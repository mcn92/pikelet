// Ingestion/anchor conformance suite: does a search result send the user
// to the correct section of the correct page? Covers markdown, MDX, and
// HTML extraction, section-aware chunking, and anchor parity with
// github-slugger (what Docusaurus renders). Part of npm test.
import { counts } from './harness.mjs';

await import('./markdown.mjs');
await import('./mdx.mjs');
await import('./html.mjs');
await import('./anchors.mjs');
await import('./strip.mjs');
await import('./folder.mjs');
await import('./git.mjs');
await import('./conversations.mjs');
await import('./short-docs.mjs');
await import('./linear.mjs');

const { passed, failed } = counts();
console.log(`\nIngestion conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
