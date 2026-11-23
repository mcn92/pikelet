// HTML ingestion conformance: existing id attributes are the anchors,
// derived slugs only as fallback, chrome elements excluded, title h1
// excluded from paths.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const CPS = path.resolve(here, '..', '..', 'packages', 'pikelet', 'src', 'ingest.mjs');
const { extractHtml, chunkDocs } = await import(pathToFileURL(CPS).href);

section('html: page.html (id attributes, fallback slugs, chrome stripping)');
{
  const html = fs.readFileSync(path.join(here, 'fixtures', 'page.html'), 'utf8');
  const extracted = extractHtml(html);
  check('page title extracted', extracted.title === 'Widgets | Example Docs', extracted.title);
  const doc = { id: 0, url: 'https://example.com/widgets', title: extracted.title, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 256, overlapPercent: 15 });
  const byAnchor = (a) => chunks.find((c) => c.anchor === a);

  const creating = byAnchor('creating-widgets');
  check('authored id attribute is the anchor', !!creating && /registering its tag name/.test(creating.text));
  check('heading path from the h2', JSON.stringify(creating?.headingPath) === '["Creating widgets"]', JSON.stringify(creating?.headingPath));

  const mount = byAnchor('widgetmount');
  check('missing id falls back to the derived slug', !!mount && /fires exactly once per instance/.test(mount.text),
    JSON.stringify(chunks.map((c) => c.anchor)));
  check('h3 nests under its h2', JSON.stringify(mount?.headingPath) === '["Creating widgets","widget:mount"]', JSON.stringify(mount?.headingPath));

  const ref = byAnchor('API_reference');
  check('authored id keeps case and underscores verbatim', !!ref && /table of contents links to/.test(ref.text));

  const intro = chunks.find((c) => /composable rendering unit/.test(c.text));
  check('h1 matching the page title stays out of paths', !!intro && intro.headingPath.length === 0, JSON.stringify(intro?.headingPath));
  check('nav and footer content never reach chunks', chunks.every((c) => !/Copyright|Home/.test(c.text)));
}

section('html: minified pages with unquoted attributes (nodejs.org style)');
{
  const { extractLinks } = await import(pathToFileURL(CPS).href);
  const minified = '<title>FS | Node.js Documentation</title><body>'
    + '<a href=documentation.html>docs</a><a href=/other>other</a>'
    + '<h2 id=fsreadfilesyncpath-options>fs.readFileSync(path[, options])</h2>'
    + '<p>Returns the contents of the path synchronously, blocking the event loop until the read completes, which is acceptable in startup code and CLI tools but not inside request handlers.</p></body>';
  const extracted = extractHtml(minified);
  const doc = { id: 0, url: 'https://nodejs.org/api/fs.html', title: extracted.title, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 256, overlapPercent: 15 });
  check('unquoted id attribute becomes the anchor',
    chunks.some((c) => c.anchor === 'fsreadfilesyncpath-options' && /blocking the event loop/.test(c.text)),
    JSON.stringify(chunks.map((c) => c.anchor)));
  const links = new Set(extractLinks(minified, 'https://nodejs.org/api/', 'https://nodejs.org'));
  check('unquoted hrefs are crawlable', links.has('https://nodejs.org/api/documentation.html') && links.has('https://nodejs.org/other'),
    JSON.stringify([...links]));
}
