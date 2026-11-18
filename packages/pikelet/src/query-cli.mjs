// pikelet query <pack> <question>: search a pack from the terminal and print
// what an MCP agent would receive, as readable text or (--json) verbatim.

import { CliError, parsePositiveInt } from './common.mjs';

const MAX_K = 20;

export async function runQueryCommand(positionals, flags) {
  const [pack, ...words] = positionals;
  const question = words.join(' ').trim();
  if (!pack || !question) {
    throw new CliError('Usage: pikelet query <pack> "<question>"\nNext: run pikelet query --help');
  }
  const k = flags.limit === undefined ? 5 : parsePositiveInt(flags.limit, '--limit');
  if (k > MAX_K) throw new CliError(`--limit must be ${MAX_K} or less`);

  const { queryOnce } = await import('./mcp.mjs');
  const { loadCompleteModules, plainHttpWarning } = await import('./common.mjs');
  const { nodeFetch } = await import('pikelet-wasm/complete/node-fetch.mjs');
  const { reader } = await loadCompleteModules();
  const warning = plainHttpWarning(pack.replace(/#[0-9a-f]{64}$/i, ''));
  if (warning) process.stderr.write(`${warning}\n`);
  let out;
  try {
    out = await queryOnce(pack, question, {
      k,
      openPikeletFile: reader.openPikeletFile,
      httpRangeSource: (url, options = {}) => reader.httpRangeSource(url, { fetch: nodeFetch, ...options }),
      blobCache: typeof reader.diskBlobCache === 'function' ? reader.diskBlobCache() : null,
    });
  } catch (err) {
    throw new CliError(err?.message || String(err));
  }
  if (flags.json) {
    console.log(JSON.stringify(out, null, 2));
    return;
  }
  console.log(formatSearch(out));
}

const VERDICT = {
  strong: 'strong: the pack covers this topic',
  weak: 'weak: the pack may cover this topic',
  none: 'none: the pack does not appear to cover this topic; results shown anyway',
  unscored: 'unscored: this pack has no calibration, so there is no relevance signal',
};

export function formatSearch(out) {
  const lines = [];
  for (const section of out.sections) {
    const confidence = typeof section.confidence === 'number' ? ` (confidence ${section.confidence.toFixed(2)})` : '';
    lines.push(`${section.pack}: ${VERDICT[section.matchQuality] ?? section.matchQuality}${confidence}`);
    if (!section.results.length) lines.push('  no results');
    section.results.forEach((r, i) => {
      const heading = [r.title, ...(r.headingPath || []).filter((h) => h !== r.title)].filter(Boolean).join(' › ');
      const source = r.source ? `${r.source}${r.anchor && !String(r.source).includes('#') ? `#${r.anchor}` : ''}` : null;
      lines.push('');
      lines.push(`${i + 1}. ${heading || `record ${r.id}`}`);
      if (source) lines.push(`   ${source}`);
      // A chunk's text opens with its own heading, already printed above.
      const last = [r.title, ...(r.headingPath || [])].filter(Boolean).pop();
      let text = String(r.text ?? '').replace(/\s+/g, ' ').trim();
      if (last && text.startsWith(`${last} `)) text = text.slice(last.length + 1);
      if (text) lines.push(`   ${text.length > 280 ? `${text.slice(0, 279)}…` : text}`);
    });
    const uncovered = section.grounding?.uncovered;
    if (Array.isArray(uncovered) && uncovered.length) {
      lines.push('');
      lines.push(`Question words the best-covering passage lacks: ${uncovered.join(', ')}`);
    }
  }
  return lines.join('\n');
}
