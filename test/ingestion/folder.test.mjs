// ingestFolder conformance: what the folder walk skips, it reports. A
// symlink (not followed) and an oversized file are both logged as warnings
// naming the path; nothing in the source tree disappears silently.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { ingestFolder, walk } = await import(pathToFileURL(path.resolve(here, '..', '..', 'packages', 'pikelet', 'src', 'ingest.mjs')).href);

section('ingestFolder: skipped entries are reported');
{
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-ingest-folder-'));
  try {
    await fs.mkdir(path.join(root, 'docs'));
    await fs.writeFile(path.join(root, 'docs', 'real.md'), '# Real\n\nA paragraph of real content that is long enough to keep.\n');
    await fs.symlink(path.join(root, 'docs', 'real.md'), path.join(root, 'docs', 'linked.md'));
    await fs.symlink(path.join(root, 'docs'), path.join(root, 'docs-link'));
    // A sparse file: the size is what the cap reads; no bytes are written.
    const big = await fs.open(path.join(root, 'docs', 'huge.md'), 'w');
    await big.truncate(16 * 1024 * 1024 + 1);
    await big.close();

    const logs = [];
    const docs = await ingestFolder(root, {}, (line) => logs.push(line));
    check('the real file is ingested', docs.length === 1 && docs[0].sourcePath === 'docs/real.md', JSON.stringify(docs.map((d) => d.sourcePath)));
    check('a symlinked file is skipped with a warning naming it', logs.some((l) => l.includes('skipped symlink docs/linked.md')), JSON.stringify(logs));
    check('a symlinked directory is skipped with a warning naming it', logs.some((l) => l.includes('skipped symlink docs-link')), JSON.stringify(logs));
    check('a file over the per-file limit is skipped with a warning naming it and the limit',
      logs.some((l) => l.includes('skipped docs/huge.md') && l.includes('per-file limit')), JSON.stringify(logs));
    check('no other warnings are emitted', logs.length === 3, JSON.stringify(logs));

    // walk() is exported and documented as walk(root); the symlink branch
    // calls log() unconditionally, so without a default the one-argument
    // form crashed on the first link it met. ingestFolder always passes a
    // log, which is why only the exported surface regressed.
    let walkThrew = null;
    try { await walk(root); } catch (error) { walkThrew = error; }
    check('walk(root) survives a symlink without a log argument', walkThrew === null,
      walkThrew ? `${walkThrew.constructor.name}: ${walkThrew.message}` : '');
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
}

section('ingestFolder: files are taken in a fixed order');
{
  // readdir order is the filesystem's (hash order on ext4), so record ids and
  // the pack identity varied between checkouts of the same tree. Files are
  // sorted by relative path, by code unit: uppercase before lowercase, and
  // '-' < '.' < '/'.
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-ingest-order-'));
  try {
    const body = (name) => `# ${name}\n\nA paragraph of content for ${name} that is long enough to keep.\n`;
    await fs.mkdir(path.join(root, 'a'));
    for (const rel of ['a/b.md', 'a.md', 'B.md', 'a-b.md', 'z.md', 'a/a.md']) {
      await fs.writeFile(path.join(root, rel), body(rel));
    }
    const docs = await ingestFolder(root, {}, () => {});
    const order = docs.map((d) => d.sourcePath);
    const want = ['B.md', 'a-b.md', 'a.md', 'a/a.md', 'a/b.md', 'z.md'];
    check('files are ingested in code-unit order of their relative paths', JSON.stringify(order) === JSON.stringify(want), JSON.stringify(order));
    check('record ids follow that order', docs.every((d, i) => d.id === i));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
}
