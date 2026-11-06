// Ingestion: folder walk and URL crawl, HTML/Markdown/MDX extraction, chunking,
// dedupe, Docusaurus route mapping, and the public chunk shape.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError } from './common.mjs';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';

const MAX_CRAWL_BODY_BYTES = 2 * 1024 * 1024;
// Per-file ceiling for folder sources, the counterpart of the crawl cap: a
// misclassified asset or a generated dump in the source tree is skipped with
// a warning instead of being read whole into the single-threaded build.
const MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024;
async function ingestFolder(root, source, log) {
  const docs = [];
  if (!fssync.existsSync(root)) {
    throw new CliError(`Source folder not found: ${root}\nNext: check --source or source.path in pikelet.config.json.`, 1);
  }
  const files = await walk(root, log);
  for (const file of files) {
    const rel = path.relative(root, file).split(path.sep).join('/');
    if (!matchesSource(rel, source)) continue;
    try {
      const { size } = await fs.stat(file);
      if (size > MAX_SOURCE_FILE_BYTES) {
        log(`warn: skipped ${rel}: ${size} bytes exceeds the ${MAX_SOURCE_FILE_BYTES}-byte per-file limit`);
        continue;
      }
      const text = await fs.readFile(file, 'utf8');
      const extracted = extractByExtension(text, file);
      if (extracted.text.trim()) docs.push({ id: docs.length, sourcePath: rel, title: extracted.title || path.basename(file), slug: extracted.slug || null, text: extracted.text, sections: extracted.sections });
    } catch (error) {
      log(`warn: skipped unreadable file ${rel}: ${error.message}`);
    }
  }
  return docs;
}

// log defaults to a no-op so the documented one-argument walk(root) works.
async function walk(root, log = () => {}, dir = root) {
  const out = [];
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch (error) {
    throw new CliError(`Unable to read source folder ${dir}: ${error.message}`, 1);
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await walk(root, log, full));
    else if (entry.isFile()) out.push(full);
    // Dirents carry lstat semantics, so a symlink is neither a file nor a
    // directory here. Links are not followed (a link out of the tree or a
    // cycle would otherwise be walked), but the skip is reported like every
    // other skip in this path instead of silently dropping the content.
    else if (entry.isSymbolicLink()) log(`warn: skipped symlink ${path.relative(root, full).split(path.sep).join('/')} (symlinks are not followed)`);
  }
  return out;
}

function matchesSource(rel, source) {
  const includes = source.include || ['**/*.{md,mdx,html,txt}'];
  const excludes = source.exclude || [];
  return includes.some((glob) => globMatch(rel, glob)) && !excludes.some((glob) => globMatch(rel, glob));
}

// A real glob->RegExp compiler, not a handful of special-cased prefixes:
// '**' matches any number of path segments (including zero), '*' matches
// within one segment, '?' matches one character, and '{a,b,c}' is
// alternation. Case-insensitive, matched full-string against the
// forward-slashed relative path.
const GLOB_CACHE = new Map();
function compileGlob(glob) {
  let cached = GLOB_CACHE.get(glob);
  if (cached) return cached;
  const normalized = glob.replace(/\\/g, '/');
  let out = '';
  for (let i = 0; i < normalized.length; i++) {
    const c = normalized[i];
    if (c === '*') {
      if (normalized[i + 1] === '*') {
        // '**/' consumes the following slash too, so it can also match zero
        // segments (e.g. '**/*.md' must match 'a.md', not just 'x/a.md').
        if (normalized[i + 2] === '/') { out += '(?:.*/)?'; i += 2; } else { out += '.*'; i += 1; }
      } else {
        out += '[^/]*';
      }
    } else if (c === '?') {
      out += '[^/]';
    } else if (c === '{') {
      const close = normalized.indexOf('}', i);
      const alts = (close === -1 ? normalized.slice(i + 1) : normalized.slice(i + 1, close))
        .split(',').map((alt) => alt.replace(/[.+^${}()|[\]\\]/g, '\\$&'));
      out += `(?:${alts.join('|')})`;
      i = close === -1 ? normalized.length : close;
    } else {
      out += c.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    }
  }
  cached = new RegExp(`^${out}$`, 'i');
  GLOB_CACHE.set(glob, cached);
  return cached;
}

function globMatch(file, glob) {
  return compileGlob(glob).test(file);
}

// Aggregate pages that duplicate a whole site's content in one document —
// crawling them alongside the per-page versions puts every chunk in the
// corpus twice (near-duplicates survive exact-text dedupe because chunk
// boundaries differ). mdBook's print.html is the known offender.
const DEFAULT_URL_EXCLUDES = ['*/print.html'];

// URL patterns are matched against the URL's pathname, full-match, with '*'
// as the only wildcard — deliberately not the filesystem glob dialect the
// folder source uses ('**', braces), which does not map onto URLs.
function urlPatternMatcher(patterns) {
  const regexes = patterns.map((pattern) => new RegExp(
    `^${pattern.split('*').map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`,
  ));
  return (pathname) => regexes.some((regex) => regex.test(pathname));
}

function urlCrawlFilter(source) {
  const excluded = urlPatternMatcher([...DEFAULT_URL_EXCLUDES, ...(source.excludeUrl || [])]);
  const included = source.includeUrl?.length ? urlPatternMatcher(source.includeUrl) : null;
  return (href) => {
    const { pathname } = new URL(href);
    if (excluded(pathname)) return false;
    return included ? included(pathname) : true;
  };
}

// A pure redirect page: HTTP 200 whose HTML is a meta refresh
// (docs.astro.build's root is `<meta http-equiv="refresh"
// content="0;url=/en/getting-started/">` and nothing else).
function metaRefreshTarget(html, baseHref) {
  const meta = html.match(/<meta[^>]*http-equiv\s*=\s*["']?refresh["']?[^>]*>/i);
  if (!meta) return null;
  const url = meta[0].match(/content\s*=\s*["']?[^"'>]*?url\s*=\s*([^"'>\s;]+)/i);
  if (!url) return null;
  try { return new URL(url[1], baseHref); } catch { return null; }
}

// The seed URL is followed through redirects (HTTP and meta refresh,
// bounded) before the crawl starts, and the final URL defines the crawl
// origin: sites often send their root to a canonical host or localized
// landing page. Mid-crawl redirects are still skipped; the crawl finds
// their targets through links.
async function resolveSeedUrl(seed, log) {
  let current = seed;
  for (let hop = 0; hop < 5; hop++) {
    let response;
    try {
      response = await nodeFetch(current.href, {
        headers: { 'User-Agent': 'pikelet/0.1.0' },
        redirect: 'manual',
        signal: AbortSignal.timeout(15000),
        // HTML only: a server that gzips despite Accept-Encoding: identity
        // still crawls (inflated by node:zlib); Brotli and zstd are refused.
        inflateGzip: true,
      });
    } catch {
      return current; // the crawl loop will surface the fetch error itself
    }
    const location = response.headers.get('location');
    if (response.status >= 300 && response.status < 400 && location) {
      await response.body?.cancel?.().catch(() => {});
      const next = new URL(location, current);
      log(`seed redirected: ${current.href} -> ${next.href}`);
      current = next;
      continue;
    }
    const type = response.headers.get('content-type') || '';
    if (response.ok && type.includes('text/html')) {
      let html = '';
      try {
        html = await readLimitedText(response, MAX_CRAWL_BODY_BYTES);
      } catch {
        return current;
      }
      const refresh = metaRefreshTarget(html, current.href);
      if (refresh && refresh.href !== current.href) {
        log(`seed redirected (meta refresh): ${current.href} -> ${refresh.href}`);
        current = refresh;
        continue;
      }
      return current;
    }
    await response.body?.cancel?.().catch(() => {});
    return current;
  }
  log(`warn: seed still redirecting after 5 hops; crawling ${current.href}`);
  return current;
}

async function ingestUrl(source, log) {
  const seed = await resolveSeedUrl(new URL(source.url), log);
  const seen = new Set();
  const queue = [seed.href];
  const docs = [];
  const allowed = urlCrawlFilter(source);
  let filtered = 0;
  while (queue.length && docs.length < (source.maxPages || 500)) {
    const href = queue.shift();
    if (seen.has(href)) continue;
    seen.add(href);
    let response;
    try {
      response = await nodeFetch(href, {
        headers: { 'User-Agent': 'pikelet/0.1.0' },
        redirect: 'manual',
        signal: AbortSignal.timeout(15000),
        inflateGzip: true,
      });
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    if (response.status >= 300 && response.status < 400) {
      log(`warn: skipped redirect from ${href}`);
      continue;
    }
    const type = response.headers.get('content-type') || '';
    if (!response.ok || !type.includes('text/html')) continue;
    let html;
    try {
      html = await readLimitedText(response, MAX_CRAWL_BODY_BYTES);
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    const extracted = extractHtml(html);
    // A contentless meta-refresh page is a redirect wearing a 200: skip it
    // as a document (it would waste page budget as an empty doc) and follow
    // its target through the normal frontier filters. Pages with real
    // content and an incidental refresh tag are kept as documents.
    const refresh = metaRefreshTarget(html, href);
    if (refresh && extracted.text.split(/\s+/).filter(Boolean).length < 25) {
      log(`warn: ${href} is a meta-refresh redirect page -> ${refresh.href}`);
      if (refresh.origin === seed.origin && !seen.has(refresh.href)) {
        if (!allowed(refresh.href)) {
          seen.add(refresh.href);
          filtered++;
        } else if (queue.length + docs.length < (source.maxPages || 500)) {
          queue.push(refresh.href);
        }
      }
      continue;
    }
    docs.push({ id: docs.length, url: href, title: extracted.title || href, text: extracted.text, sections: extracted.sections });
    // Filters gate the queue, not the dequeue: a filtered link must never
    // occupy page budget (queue.length counts toward maxPages), or a large
    // excluded section starves discovery of the pages the crawl is for.
    // The seed the user typed bypasses the filters — it seeds the queue
    // directly above.
    for (const link of extractLinks(html, href, seed.origin)) {
      if (seen.has(link)) continue;
      if (!allowed(link)) {
        seen.add(link);
        filtered++;
        continue;
      }
      if (queue.length + docs.length < (source.maxPages || 500)) queue.push(link);
    }
  }
  if (filtered > 0) {
    const filters = [
      ...(source.includeUrl?.length ? [`include ${source.includeUrl.join(' ')}`] : []),
      `exclude ${[...DEFAULT_URL_EXCLUDES, ...(source.excludeUrl || [])].join(' ')}`,
    ];
    log(`URL filters skipped ${filtered} pages (${filters.join('; ')})`);
  }
  // A crawl that discovers almost nothing usually means the site renders its
  // navigation client-side; the crawler only follows server-rendered anchors.
  if (docs.length > 0 && docs.length < 5) {
    log(`warn: crawl discovered only ${docs.length} page${docs.length === 1 ? '' : 's'} from ${seed.href}. `
      + 'If the site builds its navigation with client-side JavaScript, the crawler cannot follow it.');
  }
  return docs;
}

async function readLimitedText(response, maxBytes) {
  const declaredLength = Number(response.headers.get('content-length') || '0');
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new Error(`response body exceeds ${maxBytes} bytes`);
  }

  const reader = response.body?.getReader?.();
  if (!reader) {
    const text = await response.text();
    if (Buffer.byteLength(text, 'utf8') > maxBytes) throw new Error(`response body exceeds ${maxBytes} bytes`);
    return text;
  }

  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel().catch(() => {});
      throw new Error(`response body exceeds ${maxBytes} bytes`);
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8');
}

function extractByExtension(text, file) {
  if (/\.html?$/i.test(file)) return extractHtml(text);
  if (/\.mdx?$/i.test(file)) return extractMarkdown(text);
  text = text.replace(/^﻿/, '');
  const title = (text.match(/^#\s+(.+)$/m) || [])[1];
  return { title, text: stripMarkdown(text) };
}

// GitHub-style heading slugs, deduplicated per document with -1/-2
// suffixes, mirroring github-slugger (what Docusaurus and GitHub render):
// trim, lowercase, keep letters/numbers/space/hyphen/underscore (Unicode
// letters included), and replace EACH space with a dash without collapsing
// runs. Inline code and links are unwrapped first. Checked against the real
// github-slugger by test/ingestion/anchors.test.mjs.
function makeSlugger() {
  const used = new Map();
  return (heading) => {
    const base = String(heading || '')
      .replace(/`([^`]*)`/g, '$1')
      .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N} _-]/gu, '')
      .replace(/ /g, '-');
    const n = used.get(base) || 0;
    used.set(base, n + 1);
    return n === 0 ? base : `${base}-${n}`;
  };
}

// Strip inline markdown from a heading for display and path purposes.
// Underscores stay: snake_case identifiers (ACTION_QUERY_PARAMS) are far
// more common in developer-doc headings than _underscore emphasis_, and
// github-slugger keeps them.
const cleanHeading = (heading) => normalizeText(String(heading || '')
  .replace(/`([^`]*)`/g, '$1')
  .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
  .replace(/[*~]/g, ''));

// Split a markdown body into heading-delimited sections. Heading lines are
// recognized only outside code fences, so a `# comment` inside a fenced
// shell block never splits a section. Content before the first heading
// becomes a depth-0 preamble section.
function markdownSections(body) {
  const sections = [];
  let fence = null;
  let current = { depth: 0, heading: null, customId: null, lines: [] };
  for (const line of String(body).split('\n')) {
    const f = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (f) {
      if (!fence) fence = f[1];
      else if (f[1][0] === fence[0] && f[1].length >= fence.length) fence = null;
      current.lines.push(line);
      continue;
    }
    const h = !fence && line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (h) {
      sections.push(current);
      current = { ...headingParts(h[2], h[1].length), lines: [] };
      continue;
    }
    // Setext headings: a line of === (depth 1) or --- (depth 2, two or
    // more so a stray list dash cannot promote its neighbor) directly
    // under a non-blank text line, outside fences.
    const setext = !fence && line.match(/^\s{0,3}(=+|-{2,})\s*$/);
    if (setext) {
      const prev = current.lines[current.lines.length - 1];
      if (prev && prev.trim() && !/^\s{0,3}(#|>|[-*+]\s|\d+[.)]\s)/.test(prev)) {
        current.lines.pop();
        sections.push(current);
        current = { ...headingParts(prev.trim(), setext[1][0] === '=' ? 1 : 2), lines: [] };
        continue;
      }
    }
    current.lines.push(line);
  }
  sections.push(current);
  return sections.map((s) => ({ ...s, body: s.lines.join('\n') }));
}

// Split a raw heading into display text, depth, and an explicit id when
// present ({#custom-id}, Docusaurus/Astro style — it wins over the slug).
function headingParts(rawHeading, depth) {
  let heading = rawHeading.replace(/\s+#+\s*$/, '');
  const custom = heading.match(/\s*\{#([A-Za-z0-9_-]+)\}\s*$/);
  if (custom) heading = heading.slice(0, custom.index);
  return { depth, heading: cleanHeading(heading), customId: custom ? custom[1] : null };
}

// Assign heading paths and anchors to raw sections. The path includes the
// section's own heading; a leading h1 that carries the document title is
// structural rather than sectional and stays out of the paths.
function sectionize(rawSections, title, textOf) {
  const slugger = makeSlugger();
  const stack = [];
  const out = [];
  let sawH1Title = false;
  for (const s of rawSections) {
    if (s.depth === 0) {
      const text = textOf(s);
      if (text) out.push({ headingPath: [], anchor: '', text });
      continue;
    }
    const anchor = s.customId || slugger(s.heading);
    const isTitleH1 = s.depth === 1 && !sawH1Title && title
      && s.heading.toLowerCase() === String(title).toLowerCase();
    if (s.depth === 1 && !sawH1Title) sawH1Title = true;
    while (stack.length && stack[stack.length - 1].depth >= s.depth) stack.pop();
    const headingPath = isTitleH1 ? [] : [...stack.map((e) => e.heading).filter(Boolean), s.heading];
    stack.push({ depth: s.depth, heading: isTitleH1 ? null : s.heading });
    const text = textOf(s);
    if (text || s.heading) out.push({ headingPath, anchor, heading: s.heading, text });
  }
  return out;
}

// Extracts a single `key: value` line from YAML frontmatter. The value may
// be double-quoted, single-quoted (each with its own escapes, free to hold
// the other quote), or bare.
function frontmatterValue(frontmatter, key) {
  const re = new RegExp(`^${key}:\\s*(?:"((?:[^"\\\\]|\\\\.)*)"|'((?:[^'\\\\]|\\\\.)*)'|(.+?))\\s*$`, 'm');
  const m = frontmatter.match(re);
  if (!m) return null;
  return m[1] ?? m[2] ?? m[3] ?? null;
}

function extractMarkdown(text) {
  let body = String(text || '').replace(/^﻿/, '').replace(/\r/g, '');
  let slug = null;
  let title = null;
  const frontmatter = body.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  if (frontmatter) {
    body = body.slice(frontmatter[0].length);
    slug = frontmatterValue(frontmatter[1], 'slug');
    title = frontmatterValue(frontmatter[1], 'title');
  }
  body = body
    .replace(/^import\s[^\n]*$/gm, ' ')
    .replace(/^export\s[^\n]*$/gm, ' ')
    .replace(/<\/?[A-Z][A-Za-z0-9.]*(?:\s[^>]*?)?\/?>/g, ' ');
  const rawSections = markdownSections(body);
  // Title fallback: the first level-1 heading, whichever syntax wrote it
  // (ATX or setext) — the section parser already normalized both.
  if (!title) title = rawSections.find((s) => s.depth === 1)?.heading || null;
  // Sections carry their own heading text as the first line so retrieval
  // (embedding and lexical alike) sees what the section is about.
  const sections = sectionize(rawSections, title,
    (s) => stripMarkdown(s.heading ? `${s.heading}\n${s.body}` : s.body).trim());
  return { title, slug, text: stripMarkdown(body), sections };
}

function extractHtml(html) {
  const title = normalizeText(decodeEntities((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || ''));
  let body = (html.match(/<(main|article)[^>]*>([\s\S]*?)<\/\1>/i) || [])[2]
    || (html.match(/<body[^>]*>([\s\S]*?)<\/body>/i) || [])[1]
    || html;
  body = body.replace(/<(script|style|nav|header|footer|aside)[^>]*>[\s\S]*?<\/\1>/gi, ' ');
  // Block closers become a paragraph break (\n\n) so minified HTML — no
  // whitespace between tags — still gets paragraph-aware chunking
  // downstream (chunkDocs splits on \n{2,}); <br> is a soft line break
  // within a paragraph, so it stays a single \n. normalizeText collapses
  // any resulting run of 3+ newlines back down to exactly one blank line.
  const flattenHtml = (fragment) => normalizeText(decodeEntities(String(fragment)
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(p|li|div|tr|pre|h[1-6])>/gi, '\n\n')
    .replace(/<[^>]+>/g, ' ')));
  // Heading tags delimit sections; an id attribute on the heading (or, as
  // fallback, on an anchor/element wrapping it) is the section anchor —
  // matching what the site's own anchor links use — with a derived slug
  // when none exists.
  const raw = [];
  let cursor = 0;
  let last = { depth: 0, heading: null, customId: null };
  for (const m of body.matchAll(/<h([1-6])([^>]*)>([\s\S]*?)<\/h\1>/gi)) {
    raw.push({ ...last, body: body.slice(cursor, m.index) });
    const attrId = attrValue(m[2], 'id') || attrValue(m[3], 'id') || null;
    last = { depth: Number(m[1]), heading: normalizeText(decodeEntities(m[3].replace(/<[^>]+>/g, ' '))), customId: attrId };
    cursor = m.index + m[0].length;
  }
  raw.push({ ...last, body: body.slice(cursor) });
  const pageTitle = title.split(/\s*[|·–—-]\s+/)[0] || title;
  const sections = sectionize(raw, pageTitle,
    (s) => (s.heading ? `${s.heading}\n${flattenHtml(s.body)}` : flattenHtml(s.body)).trim());
  return { title, text: flattenHtml(body), sections };
}

// Attribute values may be unquoted (minified HTML — nodejs.org's API docs
// serve href=fs.html); accept double-quoted, single-quoted, or bare.
const attrValue = (tag, name) => {
  const m = tag.match(new RegExp(`\\b${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s"'<>=\`]+))`, 'i'));
  return m ? (m[2] ?? m[3] ?? m[4]) : null;
};

function extractLinks(html, baseHref, origin) {
  const links = [];
  for (const match of html.matchAll(/href\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'<>=`]+))/gi)) {
    try {
      const url = new URL(match[2] ?? match[3] ?? match[4], baseHref);
      url.hash = '';
      if (url.origin === origin && /^https?:$/.test(url.protocol)) links.push(url.href);
    } catch {}
  }
  return links;
}

// A fence line: optional indent, then three or more of one fence character.
// Returns the character, the run length, and whatever follows the run.
function fenceRun(line) {
  let p = 0;
  while (p < line.length && (line[p] === ' ' || line[p] === '\t')) p += 1;
  const ch = line[p];
  if (ch !== '`' && ch !== '~') return null;
  let len = 0;
  while (p + len < line.length && line[p + len] === ch) len += 1;
  if (len < 3) return null;
  return { ch, len, rest: line.slice(p + len) };
}

// Fenced code blocks, CommonMark-style: an opening fence is closed by the
// first later line holding only a fence of the same character at least as
// long; an unclosed fence runs to the end. Linear: each line is visited once.
function protectFences(src, keep) {
  const lines = src.split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const open = fenceRun(lines[i]);
    if (!open) {
      out.push(lines[i]);
      continue;
    }
    let j = i + 1;
    for (; j < lines.length; j++) {
      const close = fenceRun(lines[j]);
      if (close && close.ch === open.ch && close.len >= open.len && close.rest.trim() === '') break;
    }
    out.push(keep(`\n${lines.slice(i + 1, j).join('\n')}\n`));
    i = j;
  }
  return out.join('\n');
}

// Inline code spans, CommonMark-style: a run of N backticks is closed by
// the next run of exactly N on the same line; a run with no closer is
// literal text. Linear: each run is scanned at most twice.
function protectInlineCode(src, keep) {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const open = src.indexOf('`', i);
    if (open < 0) {
      out += src.slice(i);
      break;
    }
    out += src.slice(i, open);
    let openLen = 0;
    while (open + openLen < n && src[open + openLen] === '`') openLen += 1;
    const bodyStart = open + openLen;
    let close = -1;
    for (let k = bodyStart; k < n && src[k] !== '\n';) {
      if (src[k] !== '`') {
        k += 1;
        continue;
      }
      let runLen = 0;
      while (k + runLen < n && src[k + runLen] === '`') runLen += 1;
      if (runLen === openLen) {
        close = k;
        break;
      }
      k += runLen;
    }
    if (close < 0) {
      out += src.slice(open, bodyStart);
      i = bodyStart;
      continue;
    }
    // CommonMark strips one space from each end when both are present and
    // the body is not all spaces (`` ` `` is how a backtick is written).
    let body = src.slice(bodyStart, close);
    if (body.length >= 2 && body[0] === ' ' && body[body.length - 1] === ' ' && body.trim() !== '') {
      body = body.slice(1, -1);
    }
    out += keep(body);
    i = close + openLen;
  }
  return out;
}

// Markdown syntax is removed by position, never by deleting characters
// globally: hyphens, minus signs, underscores in identifiers, and
// comparison operators are content the record must carry verbatim.
// Fenced blocks and inline code spans pass through untouched (minus their
// delimiters); prose loses only heading hashes, blockquote/list prefixes,
// rules, boundary emphasis, table scaffolding, and link/image wrappers.
function stripMarkdown(text) {
  const src = String(text || '');
  const stash = [];
  const keep = (s) => ` ${stash.push(s) - 1} `;

  // 1. Protect code first, with the linear scanners below. Fenced blocks
  //    keep their body byte-for-byte; inline spans keep their contents.
  let out = protectInlineCode(protectFences(src, keep), keep)
    // 2. Images and links: keep alt/link text, drop the URL.
    .replace(/!\[([^\]]*)]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)]\[[^\]]*]/g, '$1')
    // 3. Block-level markers, anchored to line start.
    .replace(/^[ \t]{0,3}#{1,6}[ \t]+/gm, '')                // ATX heading hash
    .replace(/[ \t]+#+[ \t]*$/gm, '')                         // trailing closing hashes
    .replace(/^[ \t]{0,3}(?:[-*_][ \t]*){3,}$/gm, '')         // horizontal rules
    .replace(/^[ \t]{0,3}(?:>[ \t]?)+/gm, '')                 // blockquote (nested ok)
    .replace(/^([ \t]*)[-*+][ \t]+(?=\S)/gm, '$1')            // unordered list bullet
    .replace(/^([ \t]*)[-*+][ \t]+\[[ xX]][ \t]+/gm, '$1')    // task list checkbox
    // 4. Tables: drop separator rows and pipe scaffolding.
    .replace(/^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*\|?[ \t]*\n/gm, '')
    .replace(/^[ \t]*\|/gm, '').replace(/\|[ \t]*$/gm, '').replace(/[ \t]*\|[ \t]*/g, ' ')
    // 5. Inline emphasis, only at word boundaries so snake_case and
    //    intra-word asterisks survive.
    .replace(/(\*\*\*|___)(?=\S)([\s\S]*?\S)\1/g, '$2')
    .replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, '$2')
    .replace(/(?<![\w*])\*(?=\S)([^*\n]*?\S)\*(?![\w*])/g, '$1')
    .replace(/(?<![\w_])_(?=\S)([^_\n]*?\S)_(?![\w_])/g, '$1')
    .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '$1')
    // 6. Backslash escapes of markdown punctuation.
    .replace(/\\([\\`*_{}[\]()#+\-.!>~|])/g, '$1');

  // Tidy prose lines, then restore protected code verbatim (after the
  // trim, so code indentation survives).
  out = out.replace(/^[ \t]+|[ \t]+$/gm, '').replace(/[ \t]+/g, ' ');
  out = out.replace(/ (\d+) /g, (_, i) => stash[Number(i)]);
  return String(out).replace(/\r/g, '').replace(/\n{3,}/g, '\n\n').trim();
}

// Named entities beyond the load-bearing five (nbsp/amp/lt/gt/quot) that
// crawled docs and Sphinx output emit routinely — smart punctuation and a
// handful of common symbols. &amp; must decode last (its own replacement,
// '&', would otherwise get re-matched by every rule after it).
const NAMED_ENTITIES = {
  nbsp: ' ', lt: '<', gt: '>', quot: '"', apos: "'",
  mdash: '—', ndash: '–', hellip: '…',
  lsquo: '‘', rsquo: '’', ldquo: '“', rdquo: '”',
  copy: '©', reg: '®', trade: '™',
  eacute: 'é', egrave: 'è', agrave: 'à', uuml: 'ü',
  ouml: 'ö', auml: 'ä', ntilde: 'ñ', ccedil: 'ç',
};
const NAMED_ENTITY_RE = new RegExp(`&(${Object.keys(NAMED_ENTITIES).join('|')});`, 'g');

function decodeEntities(text) {
  return String(text || '')
    .replace(NAMED_ENTITY_RE, (_, name) => NAMED_ENTITIES[name])
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(Number(dec)))
    .replace(/&amp;/g, '&');
}

function normalizeText(text) {
  return String(text || '').replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
}

// Section-aware chunking: a section stays one chunk when it fits (up to
// 1.6x the target, since splitting a coherent section costs more retrieval
// quality than a modestly long chunk), subdivides at paragraph boundaries
// when it does not, and falls back to token windows only inside a single
// oversized paragraph. Every chunk carries the section's heading path and
// anchor; subdivided pieces share their section's anchor. Sections under
// 25 tokens merge into the previous chunk of the same document (their
// content is preserved; their anchor is not a retrieval target). Documents
// without sections (plain .txt) keep the flat token windows.
// The ingestion declaration (LAYERED_PROFILE.md 6.1): how records were made,
// recorded as a base's corpus.ingest so a layer cannot change record
// granularity mid-chain. Everything chunkDocs is parameterized by goes in;
// INGEST_CHUNKER names the algorithm and MUST change whenever chunkDocs or
// the extraction feeding it changes how the same source splits into records.
const INGEST_CHUNKER = 'pikelet-section-v1';
const MIN_CHUNK_TOKENS = 25;
function ingestDeclaration(chunking = {}) {
  return {
    chunker: INGEST_CHUNKER,
    targetTokens: chunking.targetTokens || 256,
    overlapPercent: chunking.overlapPercent || 15,
    minTokens: MIN_CHUNK_TOKENS,
  };
}

// The documents chunkDocs dropped for having fewer than MIN_CHUNK_TOKENS
// whitespace-separated tokens, named for a log line or an error: "a.md (12
// tokens), b.md (3 tokens)", the first `limit` of them. Empty when none were.
function describeDroppedDocs(dropped, limit = 8) {
  if (!Array.isArray(dropped) || !dropped.length) return '';
  const named = dropped.slice(0, limit)
    .map((doc) => `${doc.sourcePath || doc.url || doc.title || doc.id} (${doc.tokens} tokens)`)
    .join(', ');
  return `${named}${dropped.length > limit ? `, and ${dropped.length - limit} more` : ''}`;
}

function chunkDocs(docs, options) {
  const chunks = [];
  chunks.dropped = [];
  const { targetTokens: target, overlapPercent } = ingestDeclaration(options);
  const overlap = Math.floor(target * (overlapPercent / 100));
  for (const doc of docs) {
    const docTokens = tokenize(doc.text);
    if (docTokens.length < MIN_CHUNK_TOKENS) {
      chunks.dropped.push({
        id: doc.id,
        title: doc.title,
        sourcePath: doc.sourcePath,
        url: doc.url,
        tokens: docTokens.length,
      });
      continue;
    }
    const base = () => ({
      id: chunks.length,
      docId: doc.id,
      title: doc.title,
      headingPath: [],
      url: doc.url || doc.sourcePath,
      anchor: '',
      sourcePath: doc.sourcePath || doc.url,
      slug: doc.slug || null,
      text: '',
      // A git source's commit (sha, date, author), kept on every chunk of it.
      ...(doc.commit ? { commit: doc.commit } : {}),
      // A conversation source's exchange (conversation id, title, date, ...).
      ...(doc.conversation ? { conversation: doc.conversation } : {}),
    });
    if (!Array.isArray(doc.sections) || !doc.sections.length) {
      for (let start = 0; start < docTokens.length; start += Math.max(1, target - overlap)) {
        const slice = docTokens.slice(start, start + target);
        if (slice.length < MIN_CHUNK_TOKENS && chunks.length) {
          chunks[chunks.length - 1].text += ` ${slice.join(' ')}`;
          continue;
        }
        chunks.push({ ...base(), text: slice.join(' ') });
        if (start + target >= docTokens.length) break;
      }
      continue;
    }
    const docStart = chunks.length;
    const keepMax = Math.floor(target * 1.6);
    // A section contributes provenance only if it is headed (not a depth-0
    // preamble) and has a body beyond its heading. Either kind still
    // contributes its text to a merge.
    const hasProvenance = (section) => section.heading
      && normalizeText(section.text) !== normalizeText(section.heading);
    // Attribute a merge to the most specific location true of everything
    // merged: the longest common prefix of their heading paths, with the anchor
    // of the contributing section that owns that prefix, else '' (page level).
    // Coarse-but-correct beats precise-but-wrong for provenance.
    const attributeMerge = (contributors) => {
      const real = contributors.filter(hasProvenance);
      if (!real.length) return { headingPath: [], anchor: '' };
      let lcp = real[0].headingPath;
      for (const s of real.slice(1)) {
        const path = s.headingPath;
        let i = 0;
        while (i < lcp.length && i < path.length && lcp[i] === path[i]) i++;
        lcp = lcp.slice(0, i);
      }
      const owner = real.find((s) => s.headingPath.length === lcp.length
        && s.headingPath.every((h, i) => h === lcp[i]));
      return { headingPath: lcp, anchor: owner ? (owner.anchor || '') : '' };
    };
    const push = (section, text, contributors = [section]) => chunks.push({
      ...base(),
      ...attributeMerge(contributors),
      text: normalizeText(text),
      contributors,
    });
    // Undersized sections merge into the previous chunk, with provenance
    // re-attributed by attributeMerge. Two cases merge forward instead:
    // - an undersized section followed by its own child heading ("6. The C ABI"
    //   with a short intro, then "6.1 Handle Lifecycle") carries onto the child;
    // - a leading undersized section (often a title-only H1) has nothing before
    //   it, so it carries onto whatever comes next rather than becoming a stub
    //   that swallows the sections after it.
    const isChildOf = (parentPath, childPath) => childPath.length === parentPath.length + 1
      && parentPath.every((h, i) => h === childPath[i]);
    const sections = [];
    let carry = [];
    for (let i = 0; i < doc.sections.length; i++) {
      const raw = doc.sections[i];
      const next = doc.sections[i + 1];
      const undersized = tokenize(raw.text).length < MIN_CHUNK_TOKENS;
      if (undersized && next && isChildOf(raw.headingPath, next.headingPath)) {
        // Carry onto the child; this also covers the leading-H1 case.
        carry.push(raw);
        continue;
      }
      if (sections.length === 0 && carry.length === 0 && undersized) {
        // Leading undersized section with no child next: carry it onto the next
        // section. Later undersized sections merge backward as usual.
        carry = [raw];
        continue;
      }
      const contributors = [...carry, raw];
      carry = [];
      sections.push({ ...raw, text: contributors.map((s) => s.text).filter(Boolean).join('\n'), contributors });
    }
    if (carry.length) {
      // Every section in the document was the single leading undersized
      // one (a one-section document too small to split further); the doc
      // already cleared the 25-token floor as a whole, so it stands alone.
      sections.push({ text: carry.map((s) => s.text).filter(Boolean).join('\n'), contributors: carry });
    }
    for (const section of sections) {
      const tokens = tokenize(section.text);
      if (!tokens.length) continue;
      if (tokens.length < MIN_CHUNK_TOKENS) {
        if (chunks.length > docStart) {
          const prev = chunks[chunks.length - 1];
          prev.contributors.push(...section.contributors);
          prev.text += `\n${normalizeText(section.text)}`;
          Object.assign(prev, attributeMerge(prev.contributors));
        } else {
          push(section, section.text, section.contributors);
        }
        continue;
      }
      if (tokens.length <= keepMax) {
        push(section, section.text, section.contributors);
        continue;
      }
      let piece = [];
      let pieceTokens = 0;
      const flushPiece = () => {
        if (pieceTokens > 0) push(section, piece.join('\n\n'), section.contributors);
        piece = [];
        pieceTokens = 0;
      };
      for (const para of section.text.split(/\n{2,}/)) {
        const paraTokens = tokenize(para).length;
        if (!paraTokens) continue;
        if (paraTokens > keepMax) {
          flushPiece();
          const words = tokenize(para);
          for (let start = 0; start < words.length; start += Math.max(1, target - overlap)) {
            push(section, words.slice(start, start + target).join(' '), section.contributors);
            if (start + target >= words.length) break;
          }
          continue;
        }
        if (pieceTokens + paraTokens > target && pieceTokens > 0) flushPiece();
        piece.push(para);
        pieceTokens += paraTokens;
      }
      flushPiece();
    }
    // A document whose sections were all tiny still needs its 25-token
    // floor honored: merge a lone undersized chunk forward is impossible,
    // so it simply stays (the document itself passed the floor above).
  }
  // contributors is bookkeeping for attributeMerge's LCP computation only;
  // strip it before chunks leave chunkDocs.
  for (const c of chunks) delete c.contributors;
  return chunks;
}

function tokenize(text) {
  return normalizeText(text).split(/\s+/).filter(Boolean);
}

function dedupeChunks(chunks) {
  const seen = new Set();
  const out = [];
  for (const chunk of chunks) {
    const key = chunk.text;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ ...chunk, id: out.length });
  }
  return out;
}

function applySourceRoutes(chunks, config) {
  const baseUrl = config.source?.routeBaseUrl;
  if (!baseUrl) return;
  const routePrefix = String(config.source.routePrefix || '').replace(/^\/+|\/+$/g, '');
  const join = (...parts) => {
    const prefix = String(baseUrl || '/').endsWith('/') ? String(baseUrl || '/') : `${baseUrl}/`;
    const suffix = parts
      .map((part) => String(part || '').replace(/^\/+|\/+$/g, ''))
      .filter(Boolean)
      .join('/');
    return `${prefix}${suffix}`;
  };
  for (const chunk of chunks) {
    let route = String(chunk.sourcePath || chunk.url || '').replace(/\\/g, '/').replace(/^\.\//, '');
    route = route.replace(/\.(md|mdx|html|txt)$/i, '');
    // Docusaurus route conventions: NN- prefixes order the sidebar but are
    // stripped from routes; index/README docs become the directory route.
    route = route.split('/').map((segment) => segment.replace(/^\d+-/, '')).join('/');
    route = route.replace(/(^|\/)(index|readme)$/i, '$1').replace(/\/+$/, '');
    if (chunk.slug) {
      // Front-matter slug: absolute replaces the whole doc path (still under
      // routePrefix); relative replaces the last path segment.
      const slug = String(chunk.slug).replace(/\\/g, '/');
      route = slug.startsWith('/')
        ? slug
        : [route.split('/').slice(0, -1).join('/'), slug.replace(/^\.\//, '')].filter(Boolean).join('/');
    }
    chunk.url = join(routePrefix, route);
  }
}

function publicChunk(chunk) {
  return {
    id: chunk.id,
    docId: chunk.docId,
    title: chunk.title,
    headingPath: chunk.headingPath,
    url: chunk.url,
    anchor: chunk.anchor,
    sourcePath: chunk.sourcePath,
    text: chunk.text,
    preview: chunk.text.slice(0, 220),
    // A journal note's metadata (kind, created, authoredBy, ...), kept on the
    // record. Document chunks carry none.
    ...(chunk.note ? { note: chunk.note } : {}),
    // A git source's commit: sha, author date, author name.
    ...(chunk.commit ? { commit: chunk.commit } : {}),
    // A conversation source's exchange: which conversation, which turn, when.
    ...(chunk.conversation ? { conversation: chunk.conversation } : {}),
  };
}

// Sentence windows (an opt-in index policy, compile --sentence-windows):
// short runs of a chunk's own sentences, each indexed as an extra vector row
// that maps back to the chunk (the row map, COMPLETE_PROFILE.md 3.9). A
// chunk's single vector averages everything it says, which dilutes a
// one-sentence answer; a window keeps that sentence close to the question it
// answers. Windows carry the chunk's heading (its first line) so a sentence
// keeps the context that gives it meaning. Only the corpus's own text is used.
const SENTENCE_WINDOWS_KIND = 'sentence-windows-v1';
const SENTENCE_WINDOW_MIN_WORDS = 5;

function sentenceWindows(text, size) {
  const nl = text.indexOf('\n');
  const heading = nl === -1 ? '' : text.slice(0, nl).trim();
  const body = nl === -1 ? text : text.slice(nl + 1);
  const sentences = [];
  for (const piece of body.split(/(?<=[.!?])\s+|\n+/).map((s) => s.trim()).filter(Boolean)) {
    // Fragments under SENTENCE_WINDOW_MIN_WORDS words (list stubs, code
    // tokens) join the piece that follows them.
    const last = sentences.length - 1;
    if (last >= 0 && sentences[last].split(/\s+/).length < SENTENCE_WINDOW_MIN_WORDS) sentences[last] += ` ${piece}`;
    else sentences.push(piece);
  }
  // A chunk no longer than one window is already indexed by its own vector.
  if (sentences.length <= size) return [];
  const windows = [];
  for (let i = 0; i + size <= sentences.length; i++) {
    windows.push(`${heading}\n${sentences.slice(i, i + size).join(' ')}`);
  }
  return windows;
}

export {
  MAX_CRAWL_BODY_BYTES,
  ingestFolder,
  walk,
  matchesSource,
  globMatch,
  ingestUrl,
  resolveSeedUrl,
  urlCrawlFilter,
  readLimitedText,
  extractByExtension,
  extractMarkdown,
  extractHtml,
  extractLinks,
  stripMarkdown,
  decodeEntities,
  normalizeText,
  chunkDocs,
  describeDroppedDocs,
  MIN_CHUNK_TOKENS,
  ingestDeclaration,
  INGEST_CHUNKER,
  tokenize,
  dedupeChunks,
  applySourceRoutes,
  publicChunk,
  sentenceWindows,
  SENTENCE_WINDOWS_KIND,
};
