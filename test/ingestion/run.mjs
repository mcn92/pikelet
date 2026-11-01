// Ingestion/anchor conformance suite: does a search result send the user
// to the correct section of the correct page? Covers markdown, MDX, and
// HTML extraction, section-aware chunking, and anchor parity with
// github-slugger (what Docusaurus renders). Part of npm test.
import { counts } from './harness.mjs';

await import('./markdown.test.mjs');
await import('./mdx.test.mjs');
await import('./html.test.mjs');
await import('./anchors.test.mjs');
await import('./strip.test.mjs');
await import('./folder.test.mjs');
await import('./git.test.mjs');
await import('./conversations.test.mjs');
await import('./short-docs.test.mjs');

const { passed, failed } = counts();
console.log(`\nIngestion conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
