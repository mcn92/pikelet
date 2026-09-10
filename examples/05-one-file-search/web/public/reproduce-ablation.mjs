#!/usr/bin/env node
// Reproduces the ablation numbers from the README's "What happens if the
// answer is removed from the file?" section: veyra.pikelet and
// veyra-ablated.pikelet are byte-for-byte identical except that
// veyra-ablated.pikelet has the record "The Tovash project is housed in
// Chamber 17." removed. Same question, same encoder, same code path —
// only the file's contents differ.
//
// Usage: node examples/05-one-file-search/web/public/reproduce-ablation.mjs

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPancakeFile } from '../../../../complete/index.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const QUESTION = 'What chamber is the Tovash project housed in?';

async function run(label, file) {
    const search = await openPancakeFile(path.join(HERE, file));
    const out = await search.query(QUESTION, { k: 5 });
    const answer = out.results[0]?.preview || out.results[0]?.text || 'unsupported';
    console.log(`${label}: matchQuality: ${out.matchQuality}    confidence: ${out.confidence?.toFixed(3)}    -> ${out.results.length ? answer.split('\n')[0] : 'unsupported'}`);
    await search.close();
}

await run('full pack   ', 'veyra.pikelet');
await run('ablated pack', 'veyra-ablated.pikelet');
