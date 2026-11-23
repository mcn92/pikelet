// Markdown ingestion conformance: nested/duplicate headings, custom ids,
// fenced code, inline code/links in headings, snake_case identifiers,
// setext headings, Unicode headings and slugs, oversized-section splitting.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const CPS = path.resolve(here, '..', '..', 'packages', 'pikelet', 'src', 'ingest.mjs');
const { extractByExtension, chunkDocs, applySourceRoutes, tokenize } = await import(pathToFileURL(CPS).href);

export function pipeline(file, { routePrefix = 'docs', targetTokens = 256 } = {}) {
  const text = fs.readFileSync(path.join(here, 'fixtures', file), 'utf8');
  const extracted = extractByExtension(text, file);
  const doc = {
    id: 0,
    sourcePath: file,
    title: extracted.title || path.basename(file),
    slug: extracted.slug || null,
    text: extracted.text,
    sections: extracted.sections,
  };
  const chunks = chunkDocs([doc], { targetTokens, overlapPercent: 15 });
  applySourceRoutes(chunks, { source: { routeBaseUrl: '/', routePrefix } });
  for (const c of chunks) c.href = c.anchor ? `${c.url}#${c.anchor}` : c.url;
  return { doc, chunks };
}

const byAnchor = (chunks, anchor) => chunks.find((c) => c.anchor === anchor);

section('markdown: auth.md (nested, duplicates, custom ids, fences, inline formatting)');
{
  const { doc, chunks } = pipeline('auth.md');
  check('document title from the h1', doc.title === 'Authentication');
  check('every chunk routes to the document URL', chunks.every((c) => c.url === '/docs/auth'), JSON.stringify(chunks.map((c) => c.url)));

  const rotating = byAnchor(chunks, 'rotating-keys');
  check('nested section carries its ancestor path', JSON.stringify(rotating?.headingPath) === '["API keys","Rotating keys"]', JSON.stringify(rotating?.headingPath));
  check('nested section href', rotating?.href === '/docs/auth#rotating-keys', rotating?.href);
  check('nested section text belongs to its section', /issuing a second key first/.test(rotating?.text || ''));

  const dup = byAnchor(chunks, 'rotating-keys-1');
  check('duplicate heading gets the -1 suffix', !!dup && /slug deduplication/.test(dup.text), dup?.anchor);

  check('explicit {#custom-id} wins over the derived slug', byAnchor(chunks, 'custom-hooks')?.headingPath?.[0] === 'Webhooks');
  check('no chunk derived a webhooks slug', !byAnchor(chunks, 'webhooks'));

  const apiKeys = byAnchor(chunks, 'api-keys');
  check('fenced # comment stays inside its section, splits nothing',
    /must not become a heading/.test(apiKeys?.text || '') && !chunks.some((c) => /this comment must not/.test(c.headingPath.join(' '))));

  const config = byAnchor(chunks, 'reading-configjson-files');
  check('inline code in heading unwraps into path and slug', config?.headingPath?.[0] === 'Reading config.json files', JSON.stringify(config?.headingPath));

  const guide = byAnchor(chunks, 'see-the-upgrade-guide');
  check('link in heading slugs to its text, not its URL', guide?.headingPath?.[0] === 'See the upgrade guide', JSON.stringify(guide?.headingPath));

  const snake = byAnchor(chunks, 'action_query_params');
  check('snake_case identifier keeps its underscores', snake?.headingPath?.[0] === 'ACTION_QUERY_PARAMS', JSON.stringify(snake?.headingPath));

  const intro = chunks.find((c) => /clients prove who they are/.test(c.text));
  check('title h1 stays out of heading paths', !!intro && intro.headingPath.length === 0 && intro.anchor === 'authentication',
    JSON.stringify({ path: intro?.headingPath, anchor: intro?.anchor }));
}

section('markdown: unicode.md (setext headings, Unicode slugs)');
{
  const { doc, chunks } = pipeline('unicode.md');
  check('setext === heading becomes the document title', doc.title === 'Guía de la API', doc.title);
  const uber = byAnchor(chunks, 'über-uns');
  check('setext --- heading is depth 2 with a Unicode slug', uber?.headingPath?.[0] === 'Über uns', JSON.stringify(uber?.headingPath));
  const jp = byAnchor(chunks, '日本語の見出し');
  check('fully Japanese heading keeps every character in its anchor', !!jp && /Unicode letters and numbers/.test(jp.text));
  check('a dashed line under a list item is not promoted to a heading',
    /cannot be promoted/.test(jp?.text || '') && !chunks.some((c) => c.headingPath.includes('- not a heading')));
}

section('markdown: oversized sections split at paragraph boundaries');
{
  const paragraphs = Array.from({ length: 30 }, (_, i) =>
    `Paragraph ${i} of the long section explains one more of the many configuration values in enough words to count as a real paragraph for splitting purposes here.`);
  const text = `# Long Doc\n\nIntro paragraph long enough to stand as its own leading chunk of the long document under test today, with sufficient words to pass the floor.\n\n## The long section\n\n${paragraphs.join('\n\n')}\n`;
  const file = 'long.md';
  const extracted = extractByExtension(text, file);
  const doc = { id: 0, sourcePath: file, title: extracted.title, slug: null, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 128, overlapPercent: 15 });
  const pieces = chunks.filter((c) => c.anchor === 'the-long-section');
  check('oversized section splits into multiple chunks', pieces.length >= 3, `${pieces.length} pieces`);
  check('every piece shares the section anchor and path',
    pieces.every((c) => c.headingPath.join('>') === 'The long section'));
  check('no piece exceeds the keep-together ceiling',
    pieces.every((c) => tokenize(c.text).length <= Math.floor(128 * 1.6)),
    JSON.stringify(pieces.map((c) => tokenize(c.text).length)));
  check('paragraphs are not split mid-sentence across pieces',
    pieces.every((c) => /here\.$/.test(c.text.trim()) || /section$/.test(c.text.trim())));
}

section('markdown: no-intro.md (h1 immediately followed by h2, no intro paragraph)');
{
  // Regression: a leading section under 25 tokens (typically a title-only
  // H1 with no intro paragraph) has no previous chunk to merge backward
  // into like every other undersized section does, so it used to fall
  // through and become its own title-only chunk — which then also
  // swallowed the next undersized section (its real content, wrong
  // anchor), since *that* section's backward-merge target was the stub
  // instead of whatever comes after it.
  const { doc, chunks } = pipeline('no-intro.md');
  check('document title is still the h1 text', doc.title === 'Snapshot restore', doc.title);
  check('no chunk is a title-only stub',
    !chunks.some((c) => c.headingPath.length === 0 && c.text.trim() === 'Snapshot restore'),
    JSON.stringify(chunks.map((c) => ({ headingPath: c.headingPath, text: c.text.slice(0, 40) }))));
  check('the h1 stub does not rank as its own retrievable record',
    !chunks.some((c) => c.anchor === 'snapshot-restore'), JSON.stringify(chunks.map((c) => c.anchor)));

  const refunds = chunks.find((c) => /Refunds are issued/.test(c.text));
  check('the short leading section (Refunds) merged forward, not lost',
    !!refunds, JSON.stringify(chunks.map((c) => c.text.slice(0, 30))));
  // The H1 stub is heading-only (no body of its own — sectionize()'s text
  // is just the heading line), so it contributes no provenance: Refunds is
  // the only real content in this chunk, and gets its own anchor exactly
  // as if the stub were not there at all. This is the precise case where
  // precision is correct, unlike the ambiguous multi-sibling case below.
  check('it carries its own anchor — the h1 stub contributes no provenance',
    refunds?.anchor === 'refunds' && refunds.headingPath.join('>') === 'Refunds',
    JSON.stringify({ anchor: refunds?.anchor, headingPath: refunds?.headingPath }));

  const overview = byAnchor(chunks, 'overview');
  check('the next, normal-sized section is unaffected and stands alone',
    !!overview && /A snapshot captures the full index state/.test(overview.text));
  const compat = byAnchor(chunks, 'compatibility');
  check('a later, normal-sized section is unaffected', !!compat && /is refused rather than allowed/.test(compat.text));
}

section('markdown: ambiguous multi-sibling merges attribute to the page, not a wrong sibling');
{
  // A second regression this same fix must not reintroduce: when several
  // undersized sections under *different* headings merge into one chunk,
  // no single one of them describes the whole chunk. Attributing the merge
  // to whichever section happened to be first (forward-carry) or last
  // (backward-merge) to join is wrong-but-precise — it points a query
  // whose match came from one sibling at a URL fragment for a different
  // one. Coarse-but-correct (the page, no fragment) is the right call for
  // a project whose thesis is that a result identifies the exact evidence
  // location: a wrong precise answer is worse than an honest coarse one.
  const text = [
    '# Config reference', '',
    '## Timeout', '', 'Default timeout is 30 seconds unless overridden per request.', '',
    '## Retries', '', 'Failed requests retry up to three times with backoff.', '',
    '## Backoff', '', 'Backoff starts at 200ms and doubles each retry.', '',
    '## Pooling', '', 'Connections are pooled per host, up to 16 concurrent.', '',
    '## Logging', '', 'Request logs include method, path, and status code.', '',
  ].join('\n');
  const file = 'config-reference.md';
  const extracted = extractByExtension(text, file);
  const doc = { id: 0, sourcePath: file, title: extracted.title, slug: null, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 256, overlapPercent: 15 });
  check('every undersized sibling section merges into one chunk', chunks.length === 1, JSON.stringify(chunks.map((c) => c.text.length)));
  check('the merged chunk has no fragment anchor (page-level, not a wrong sibling)',
    chunks[0].anchor === '' && chunks[0].headingPath.length === 0,
    JSON.stringify({ anchor: chunks[0].anchor, headingPath: chunks[0].headingPath }));
  check('none of the sibling content is lost',
    /Default timeout/.test(chunks[0].text) && /Backoff starts/.test(chunks[0].text) && /status code/.test(chunks[0].text));
}

section('markdown: backward merge into a normal-sized section does not inherit its anchor for an unrelated sibling');
{
  // A pre-existing bug with the same root cause as the two regressions
  // above, just via the other merge direction: a normal-sized section
  // (Serialize) already stood as its own chunk, then a later undersized
  // sibling under a *different* heading (Compaction before export) merged
  // backward into it. The old behavior kept Serialize's own anchor for the
  // combined chunk even though the merged-in content isn't about
  // serializing — same wrong-but-precise failure, now fixed by attributing
  // every merge (forward or backward) via the longest common heading-path
  // prefix of everything actually merged in.
  const para = 'A snapshot serializes the graph, vectors, and deletion markers into one contiguous buffer that a compatible reader can restore from directly. ';
  const text = `# Snapshots\n\n## Serialize\n\n${para.repeat(4)}\n\n## Compaction before export\n\nGhosts must be compacted before export or the snapshot silently resurrects them.\n`;
  const file = 'snapshots-merge.md';
  const extracted = extractByExtension(text, file);
  const doc = { id: 0, sourcePath: file, title: extracted.title, slug: null, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 256, overlapPercent: 15 });
  check('Serialize and the short Compaction sibling merge into one chunk', chunks.length === 1, JSON.stringify(chunks.map((c) => c.text.length)));
  check('the merged chunk does not keep Serialize\'s own anchor for content that is not about serializing',
    chunks[0].anchor === '' && chunks[0].headingPath.length === 0,
    JSON.stringify({ anchor: chunks[0].anchor, headingPath: chunks[0].headingPath }));
  check('both sections\' content survive the merge',
    /serializes the graph/.test(chunks[0].text) && /compacted before export/.test(chunks[0].text));
}

section('markdown: merges under a real, shared parent section stay precisely attributed');
{
  // The LCP rule should not over-correct into always landing on the page:
  // when merged sections really do share a common ancestor that itself has
  // a body, the merge is precisely and correctly attributed to that
  // ancestor, not forced to page-level just because a merge happened.
  const para = 'Authentication covers how clients prove identity before any request is accepted by the service layer. ';
  const text = `# API\n\n## Authentication\n\n${para.repeat(3)}\n\n### Rotating keys\n\nRotate keys via the dashboard.\n\n### Revoking keys\n\nRevoke immediately invalidates the key everywhere.\n`;
  const file = 'api-nested-merge.md';
  const extracted = extractByExtension(text, file);
  const doc = { id: 0, sourcePath: file, title: extracted.title, slug: null, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 256, overlapPercent: 15 });
  check('all three sections merge into one chunk', chunks.length === 1, JSON.stringify(chunks.map((c) => c.text.length)));
  check('the merge is attributed to the shared parent, which really does cover all of it',
    chunks[0].anchor === 'authentication' && chunks[0].headingPath.join('>') === 'Authentication',
    JSON.stringify({ anchor: chunks[0].anchor, headingPath: chunks[0].headingPath }));
}
