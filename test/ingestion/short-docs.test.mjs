// What ingest drops, it names: a document under MIN_CHUNK_TOKENS is skipped
// whole, and compile and append log which ones (describeDroppedDocs), so a
// short page never goes missing from a pack without a word. The
// calibration-question checks live here too: a human-written question that
// names no record, or a questions file that does not exist, is reported
// rather than dropped silently.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = (name) => pathToFileURL(path.resolve(here, '..', '..', 'packages', 'pikelet', 'src', name)).href;
const { chunkDocs, describeDroppedDocs, MIN_CHUNK_TOKENS } = await import(src('ingest.mjs'));
const { calibrationQuestionUnmatched, loadCalibrationQueries } = await import(src('calibrate.mjs'));

section('ingest: short documents are named');
{
  const words = (n) => Array.from({ length: n }, (_, i) => `word${i}`).join(' ');
  const docs = [
    { id: 0, title: 'Long', sourcePath: 'long.md', text: words(MIN_CHUNK_TOKENS + 10) },
    { id: 1, title: 'Short', sourcePath: 'short.md', text: words(3) },
  ];
  const chunked = chunkDocs(docs, { targetTokens: 256, overlapPercent: 0 });
  check('the long document is kept and the short one dropped',
    chunked.length === 1 && chunked.dropped.length === 1 && chunked.dropped[0].sourcePath === 'short.md');
  check('the dropped document is named with its token count', describeDroppedDocs(chunked.dropped) === 'short.md (3 tokens)',
    describeDroppedDocs(chunked.dropped));
  check('nothing dropped describes as empty', describeDroppedDocs([]) === '' && describeDroppedDocs(undefined) === '');
  const many = Array.from({ length: 11 }, (_, i) => ({ sourcePath: `d${i}.md`, tokens: i }));
  check('a long list names the first eight and counts the rest',
    describeDroppedDocs(many).endsWith('d7.md (7 tokens), and 3 more') && !describeDroppedDocs(many).includes('d8.md'),
    describeDroppedDocs(many));
}

section('calibration: questions that name no record are reported');
{
  const chunks = [{ title: 'How a .pikelet query runs' }, { title: 'Snapshot restore' }];
  check('a title that matches a record passes', calibrationQuestionUnmatched({ sourceTitle: 'Snapshot restore' }, chunks) === null);
  check('an id inside the corpus passes', calibrationQuestionUnmatched({ sourceId: 1 }, chunks) === null);
  const formatted = calibrationQuestionUnmatched({ sourceTitle: 'How a `.pikelet` query runs' }, chunks);
  check('a title written with Markdown formatting is named, with the stripped title suggested',
    /no record is titled/.test(formatted || '') && formatted.includes('did you mean "How a .pikelet query runs"'), formatted);
  const missing = calibrationQuestionUnmatched({ sourceTitle: 'Nothing like it' }, chunks);
  check('an unknown title is named without a suggestion', /no record is titled "Nothing like it"$/.test(missing || ''), missing);
  check('an out-of-range id is named', /expectId 7 is not a record id \(0-1\)/.test(calibrationQuestionUnmatched({ sourceId: 7 }, chunks) || ''));

  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-calq-'));
  try {
    let err = null;
    try {
      await loadCalibrationQueries({ runtime: { calibrationQueries: 'nope.jsonl' } }, dir, () => {});
    } catch (e) { err = e; }
    check('a questions file that does not exist is an error naming it, not an empty list',
      /calibration queries file not found: .*nope\.jsonl/.test(err?.message || ''), err?.message);
    await fs.writeFile(path.join(dir, 'q.jsonl'), `${JSON.stringify({ text: 'how do restores work', expectTitle: 'Snapshot restore' })}\n`);
    const loaded = await loadCalibrationQueries({ runtime: { calibrationQueries: 'q.jsonl' } }, dir, () => {});
    check('an existing questions file loads', loaded.length === 1 && loaded[0].sourceTitle === 'Snapshot restore');
    check('no questions file configured is still an empty list', (await loadCalibrationQueries({}, dir, () => {})).length === 0);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}
