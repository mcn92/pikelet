// Golden queries: questions a pack carries inside itself, each verified at
// build time to retrieve its source, so verify_pack (and any reader) can
// re-run them against the shipped bytes. Author-written questions
// (compile --golden-queries) come first; the rest are generated from the
// corpus: title templates ("what is <title>") and runs of a chunk's content
// words, half each.

import { titleQuestions, contentWordQuestions, loadCalibrationQueries, calibrationQuestionUnmatched } from './calibrate.mjs';

export const MAX_GOLDENS = 24;
// Generated candidates per build; most verify, and the kept ones are spread
// evenly across both generators.
const CANDIDATES = 96;
const SEED = 424242;
// A golden passes when its source is among the top results at this depth,
// the depth verify_pack replays at.
export const GOLDEN_K = 10;

const reproduces = (golden, results) => (golden.expectId !== undefined
  ? results.some((r) => r.id === golden.expectId)
  : results.some((r) => (r.title || '').trim() === golden.expectTitle));

/**
 * Pick the golden queries for a pack. `query(text)` is the assembled pack's
 * own retrieval (reader query() at GOLDEN_K), so a golden is true of the
 * shipped file by construction. Returns { goldens, authored, authoredKept,
 * generated, generatedKept }.
 */
export async function pickGoldenQueries({ chunks, config, projectDir, query, log = () => {} }) {
  const toGolden = ({ text, sourceId, sourceTitle }) => ({
    text, ...(sourceId !== undefined ? { expectId: sourceId } : { expectTitle: sourceTitle }),
  });
  const authoredKept = [];
  const authored = await loadCalibrationQueries(config, projectDir, log);
  for (const q of authored) {
    const unmatched = calibrationQuestionUnmatched(q, chunks);
    if (unmatched) {
      log(`Golden queries: skipping ${JSON.stringify(q.text.slice(0, 80))}: ${unmatched}`);
      continue;
    }
    const golden = toGolden(q);
    if (reproduces(golden, await query(q.text))) authoredKept.push(golden);
    else log(`Golden queries: ${JSON.stringify(q.text.slice(0, 80))} does not retrieve its source in the top ${GOLDEN_K}; not embedded`);
  }

  const titles = [...new Set(chunks.map((c) => (c.title || '').trim()).filter(Boolean))];
  const candidates = [
    ...(titles.length ? titleQuestions(titles, Math.ceil(CANDIDATES / 2), SEED ^ 0x51f15e) : []),
    ...contentWordQuestions(chunks, Math.floor(CANDIDATES / 2), SEED ^ 0xc0ffee),
  ];
  const generatedKept = [];
  for (const c of candidates) {
    const golden = toGolden(c);
    if (reproduces(golden, await query(c.text))) generatedKept.push(golden);
  }
  // Authored questions are never displaced; generated ones fill the rest,
  // spread evenly so both generators are represented.
  const rest = Math.max(0, MAX_GOLDENS - authoredKept.length);
  const spread = generatedKept.length <= rest
    ? generatedKept
    : Array.from({ length: rest }, (_, i) => generatedKept[Math.floor(i * (generatedKept.length / rest))]);
  return {
    goldens: [...authoredKept.slice(0, MAX_GOLDENS), ...spread].slice(0, MAX_GOLDENS),
    authored: authored.length,
    authoredKept: authoredKept.length,
    generated: candidates.length,
    generatedKept: generatedKept.length,
  };
}
