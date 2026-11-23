// The text scanners in packages/pikelet/src/text-scan.mjs replace regular
// expressions that backtrack on long runs of spaces, brackets, '<' or '/'.
// Each scanner must return exactly what its expression returns (checked on
// random input built from the characters that matter to it), and must stay
// fast on the long runs that made the expression quadratic.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, '..', '..', 'packages', 'pikelet', 'src');
const T = await import(pathToFileURL(path.join(src, 'text-scan.mjs')).href);
const { stripMarkdown, extractMarkdown, extractHtml, extractByExtension, parseRobots } = await import(pathToFileURL(path.join(src, 'ingest.mjs')).href);

let seed = 0x5eed;
const random = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 2 ** 32; };
const generate = (alphabet, max) => {
  let s = '';
  for (let n = Math.floor(random() * max); n > 0; n--) s += alphabet[Math.floor(random() * alphabet.length)];
  return s;
};

const ANY = ['#', ' ', '\t', '\n', '\r', ' ', ' ', '[', ']', '(', ')', '!', '|', '-', ':', '{', '}', '<', '>', '/',
  'a', 'u', 'r', 'l', '=', '"', "'", ';', '·', '—', 'x', 'url=', 'content=', '{#id}', ' # ', '|--|', ':-:'];
const LINKS = ['[', ']', '(', ')', '!', 'a', ' ', '\n', '[a]', '](', '[]'];
const TABLE = ['|', '-', ':', ' ', '\t', '\n', '\r', 'a', '|-', '--|', '| --- |'];
const PIPES = ['|', ' ', '\t', 'a', '\n'];
const TAGS = ['<', '>', 'a', ' ', '<>'];
const oneLine = (s) => s.replace(/\n/g, '');
const asJson = (f) => (s) => JSON.stringify(f(s));

const cases = [
  ['trimChar', ANY, (s) => T.trimChar(s, '/'), (s) => s.replace(/^\/+|\/+$/g, '')],
  ['trimCharEnd', ANY, (s) => T.trimCharEnd(s, '/'), (s) => s.replace(/\/+$/, '')],
  ['replaceTags', TAGS, (s) => T.replaceTags(s, ' '), (s) => s.replace(/<[^>]+>/g, ' ')],
  ['unwrapLinks image', LINKS, (s) => T.unwrapLinks(s, 'image'), (s) => s.replace(/!\[([^\]]*)]\([^)]*\)/g, '$1')],
  ['unwrapLinks inline', LINKS, (s) => T.unwrapLinks(s, 'inline'), (s) => s.replace(/\[([^\]]+)]\([^)]*\)/g, '$1')],
  ['unwrapLinks reference', LINKS, (s) => T.unwrapLinks(s, 'reference'), (s) => s.replace(/\[([^\]]+)]\[[^\]]*]/g, '$1')],
  ['spacePipes', PIPES, T.spacePipes, (s) => s.replace(/[ \t]*\|[ \t]*/g, ' ')],
  ['trimSpaceTab per line', ANY, (s) => T.mapLines(s, T.trimSpaceTab), (s) => s.replace(/^[ \t]+|[ \t]+$/gm, '')],
  ['stripClosingHashLine', ANY, (s) => T.mapLines(s, T.stripClosingHashLine), (s) => s.replace(/[ \t]+#+[ \t]*$/gm, '')],
  ['dropTableSeparatorRows', TABLE, T.dropTableSeparatorRows,
    (s) => s.replace(/^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*\|?[ \t]*\n/gm, '')],
  ['refreshContentUrl', ANY, T.refreshContentUrl,
    (s) => (s.match(/content\s*=\s*["']?[^"'>]*?url\s*=\s*([^"'>\s;]+)/i) || [null, null])[1]],
  ['robotsField', ANY, (s) => JSON.stringify(T.robotsField(oneLine(s))),
    (s) => { const m = /^([A-Za-z-]+)\s*:\s*(.*)$/.exec(oneLine(s)); return JSON.stringify(m ? [m[1], m[2]] : null); }],
  ['firstAtxTitle', ANY, T.firstAtxTitle, (s) => (s.match(/^#\s+(.+)$/m) || [])[1]],
  ['atxHeading', ANY, (s) => JSON.stringify(T.atxHeading(oneLine(s))),
    (s) => { const m = oneLine(s).match(/^(#{1,6})\s+(.+?)\s*$/); return JSON.stringify(m ? { depth: m[1].length, text: m[2] } : null); }],
  ['stripClosingHashes', ANY, T.stripClosingHashes, (s) => s.replace(/\s+#+\s*$/, '')],
  ['customIdSuffix', ANY, asJson(T.customIdSuffix),
    (s) => { const m = s.match(/\s*\{#([A-Za-z0-9_-]+)\}\s*$/); return JSON.stringify(m ? { index: m.index, id: m[1] } : null); }],
  ['titleBeforeSiteName', ANY, T.titleBeforeSiteName, (s) => s.split(/\s*[|·–—-]\s+/)[0]],
];

section('text scanners: same result as the expression they replace');
for (const [name, alphabet, mine, expression] of cases) {
  let mismatch = null;
  for (let i = 0; i < 20000 && !mismatch; i++) {
    // Half the inputs start like the shape being parsed, so the expression
    // matches often rather than almost never.
    const lead = i % 2 ? ['#', '# ', '##\t', 'content=', 'a:', '|', '![', '['][i % 8] : '';
    const s = lead + generate(alphabet, 24);
    if (mine(s) !== expression(s)) mismatch = s;
  }
  check(name, mismatch === null, mismatch === null ? '' : `differs on ${JSON.stringify(mismatch)}`);
}

section('text scanners: linear on long runs');
const RUN = 200_000;
const timed = (label, f) => {
  const start = performance.now();
  f();
  const ms = performance.now() - start;
  // The replaced expressions take minutes on these inputs; a linear scan
  // takes milliseconds, so the bound is generous for slow CI runners.
  check(`${label} (${ms.toFixed(0)} ms)`, ms < 2000);
};
const spaces = ' '.repeat(RUN);
timed('stripMarkdown: spaces before a non-pipe, non-hash', () => stripMarkdown(`a${spaces}x\n${spaces}x\n|${'-|'.repeat(RUN / 2)}x`));
timed('stripMarkdown: unclosed brackets', () => stripMarkdown(`${'['.repeat(RUN)}\n${'![a]'.repeat(RUN / 4)}\n${'[a]['.repeat(RUN / 4)}`));
timed('extractMarkdown: heading with long whitespace', () => extractMarkdown(`# a${spaces}\u0000\n## b${spaces}{#x\n### c${spaces}x`));
timed('extractByExtension: title line of whitespace', () => extractByExtension(`#${spaces}`, 'notes.txt'));
timed('extractHtml: unclosed tags and title spaces', () => extractHtml(`<title>a${spaces}x</title><h1>${'<'.repeat(RUN)}</h1>${'<'.repeat(RUN)}`));
timed('parseRobots: field with long whitespace', () => parseRobots(`User-agent:${spaces} x\nDisallow:${spaces}/`));
timed('refreshContentUrl: repeated content=', () => T.refreshContentUrl(`<meta ${'content='.repeat(RUN / 8)}>`));
timed('trimChar: inner run of slashes', () => T.trimChar(`a${'/'.repeat(RUN)}x`, '/'));
