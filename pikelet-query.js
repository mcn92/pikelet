import { openPancakeFile } from 'pikelet-wasm/complete';

const file = process.argv[2];
const query = process.argv[3];

if (!file || !query) {
  console.error('usage: node pikelet-query.mjs <file.pikelet> "<query>"');
  process.exit(1);
}

const reader = await openPancakeFile(file);
console.log(reader.info());

const result = await reader.query(query, { k: 3 });
console.log(JSON.stringify(result, null, 2));

await reader.close();