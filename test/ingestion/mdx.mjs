// MDX ingestion conformance: frontmatter title/slug, imports/exports and
// JSX components around headings, explicit heading ids.
import { check, section } from './harness.mjs';
import { pipeline } from './markdown.mjs';

section('mdx: guide.mdx (frontmatter, components, explicit ids)');
{
  const { doc, chunks } = pipeline('guide.mdx');
  check('frontmatter title wins', doc.title === 'Getting Started', doc.title);
  check('frontmatter slug drives the route', chunks.every((c) => c.url === '/docs/start-here'), chunks[0]?.url);

  const intro = chunks.find((c) => /between MDX component blocks/.test(c.text));
  check('intro under the title h1 has an empty heading path', !!intro && intro.headingPath.length === 0, JSON.stringify(intro?.headingPath));
  check('imports, exports, and JSX wrappers are gone from chunk text',
    chunks.every((c) => !/^import |export const answer|<Tabs|<TabItem|<CodeBlock|@theme/m.test(c.text)),
    JSON.stringify(chunks.map((c) => c.text.slice(0, 40))));

  const install = chunks.find((c) => c.anchor === 'install');
  check('heading after a JSX component sections normally',
    install?.headingPath?.[0] === 'Install' && /package manager of choice/.test(install.text));
  check('component content inside a section keeps its text', /npm create thing@latest/.test(install?.text || ''));

  const configure = chunks.find((c) => c.anchor === 'configuration');
  check('explicit {#configuration} id is the anchor', !!configure && configure.headingPath[0] === 'Configure',
    JSON.stringify({ anchor: configure?.anchor, path: configure?.headingPath }));
  check('final href composes url and explicit id', configure?.href === '/docs/start-here#configuration', configure?.href);
}
