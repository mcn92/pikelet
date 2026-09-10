#!/usr/bin/env node
// Convert a downloaded BEIR dataset (bench/beir/cache/<dataset>/) into Pikelet
// source records (bench/beir/work/<dataset>/records.jsonl) plus an id map
// (bench/beir/work/<dataset>/id-map.json) from Pikelet row index -> BEIR id.
//
// One BEIR corpus document -> exactly one Pikelet record. Text is built
// deterministically as `<title>\n\n<body>`. No rechunking, no model
// involvement — this is a pure, positional ID-preserving transform.
//
// Usage: node bench/beir/convert.mjs <dataset>

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];

if (!dataset) {
  console.error('usage: node bench/beir/convert.mjs <dataset>');
  process.exit(1);
}

const cacheDir = path.join(__dirname, 'cache', dataset);
const workDir = path.join(__dirname, 'work', dataset);

if (!existsSync(cacheDir)) {
  console.error(`${cacheDir} not found — run: python3.11 bench/beir/download.py ${dataset}`);
  process.exit(1);
}

mkdirSync(workDir, { recursive: true });

async function readJsonl(filePath) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (trimmed) rows.push(JSON.parse(trimmed));
  }
  return rows;
}

function buildText(title, body) {
  const t = (title || '').trim();
  const b = (body || '').trim();
  return t ? `${t}\n\n${b}` : b;
}

const corpus = await readJsonl(path.join(cacheDir, 'corpus.jsonl'));
const queries = await readJsonl(path.join(cacheDir, 'queries.jsonl'));

// Deterministic order: as read from corpus.jsonl. pikeletRow is the position
// in this array, fixed at conversion time and reused by build.mjs.
const records = [];
const idMap = []; // idMap[pikeletRow] = beirId

for (let i = 0; i < corpus.length; i++) {
  const doc = corpus[i];
  records.push({
    _id: doc._id,
    title: doc.title || '',
    text: buildText(doc.title, doc.text),
  });
  idMap.push({ pikeletRow: i, beirId: doc._id });
}

writeFileSync(
  path.join(workDir, 'records.jsonl'),
  records.map((r) => JSON.stringify(r)).join('\n') + '\n'
);
writeFileSync(path.join(workDir, 'id-map.json'), JSON.stringify(idMap, null, 2));

// Queries pass through unchanged — query.mjs reads queries.jsonl directly
// from cache/, and filters to whichever ids appear in qrels/test.tsv.
writeFileSync(
  path.join(workDir, 'queries.jsonl'),
  queries.map((q) => JSON.stringify(q)).join('\n') + '\n'
);

console.log(
  `${dataset}: ${records.length} corpus records, ${queries.length} queries -> ${workDir}`
);
