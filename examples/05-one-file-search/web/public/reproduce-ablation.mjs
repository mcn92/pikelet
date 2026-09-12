#!/usr/bin/env node
// Reproduces the ablation and mutation numbers from the README's "What
// happens if the answer is changed or removed from the file?" section.
// veyra.pikelet, veyra-ablated.pikelet, and veyra-chamber43.pikelet are
// byte-for-byte identical except for one source record ("The Tovash
// project is housed in Chamber 17."): veyra-ablated.pikelet has that
// record removed, veyra-chamber43.pikelet has it edited to say Chamber 43
// instead. Same question, same encoder, same code path — only the file's
// contents differ. (This reproduces the retrieval side only — matchQuality,
// confidence, and the retrieved text. The README's paired LLM-session
// claims were run separately and aren't scripted here.)
//
// Usage: node examples/05-one-file-search/web/public/reproduce-ablation.mjs

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPikeletFile } from '../../../../complete/index.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const QUESTION = 'What chamber is the Tovash project housed in?';

async function run(label, file) {
    const search = await openPikeletFile(path.join(HERE, file));
    const out = await search.query(QUESTION, { k: 5 });
    const answer = out.results[0]?.preview || out.results[0]?.text || 'unsupported';
    console.log(`${label}: matchQuality: ${out.matchQuality}    confidence: ${out.confidence?.toFixed(3)}    -> ${out.results.length ? answer.split('\n')[0] : 'unsupported'}`);
    await search.close();
}

await run('full pack     ', 'veyra.pikelet');
await run('ablated pack  ', 'veyra-ablated.pikelet');
await run('chamber43 pack', 'veyra-chamber43.pikelet');
