import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PikeletMcpClient } from './mcp_client.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PACK = path.join(__dirname, 'modest-proposal.pikelet');

const QUERIES = [
  {
    label: 'Straightforward',
    question: 'What does the author propose doing with poor children?',
    note: 'A direct question about the essay\'s stated content. Retrieval should surface the proposal itself with strong support.',
  },
  {
    label: 'False premise (adversarial)',
    question: 'What farming techniques does Swift recommend for raising healthier children to eat?',
    note:
      "This assumes the essay is a sincere farming manual. It is satire: Swift's actual point is the " +
      "opposite of the literal proposal. Retrieval still surfaces passages that describe the proposal in " +
      "concrete, confident-sounding detail (yields, culinary uses) — because that content is genuinely in " +
      "the corpus. matchQuality measures whether the pack covers the topic, and it does, so it does not " +
      "flag the false premise either; only reading the passage does.",
  },
];

async function main() {
  const client = new PikeletMcpClient({ packs: [PACK] });
  await client.initialize();

  const { packs } = await client.call('list_packs');
  console.log('Mounted pack(s):');
  for (const p of packs) {
    console.log(`  ${p.name}  records=${p.records}  identity=${p.identity.slice(0, 12)}...`);
  }
  console.log();

  for (const { label, question, note } of QUERIES) {
    console.log(`=== ${label} ===`);
    console.log(`Q: ${question}`);
    const result = await client.call('search', { query: question, k: 3 });
    const section = Array.isArray(result?.results) ? result : result?.sections?.[0];
    console.log(`matchQuality: ${section?.matchQuality}   confidence: ${section?.confidence}`);
    for (const r of section?.results ?? []) {
      const preview = (r.text || '').replace(/\s+/g, ' ').slice(0, 160);
      console.log(`  [${r.distance?.toFixed(4)}] ${r.title} — ${preview}...`);
    }
    console.log(`Note: ${note}`);
    console.log();
  }

  const verdict = await client.call('verify_pack', { pack: 'modest-proposal' });
  console.log('verify_pack:', JSON.stringify(verdict, null, 2));

  client.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
