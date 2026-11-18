// usage: node scripts/stage-package-docs.mjs <package-dir-name> [--clean]
// npm prepack/postpack helper: copies the repository's CHANGELOG.md,
// FORMAT.md, LICENSE, NOTICE and README.md into packages/<name>/ so the
// published tarball carries them without the repo keeping duplicate copies;
// --clean removes them again.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [name, flag] = process.argv.slice(2);
if (!name) {
  console.error('usage: node scripts/stage-package-docs.mjs <package-dir-name> [--clean]');
  process.exit(2);
}
const target = path.join(ROOT, 'packages', name);
if (!fs.existsSync(path.join(target, 'package.json'))) {
  console.error(`no package at ${target}`);
  process.exit(2);
}
const FILES = ['CHANGELOG.md', 'FORMAT.md', 'LICENSE', 'NOTICE', 'README.md'];
for (const file of FILES) {
  const dest = path.join(target, file);
  if (flag === '--clean') {
    fs.rmSync(dest, { force: true });
  } else {
    fs.copyFileSync(path.join(ROOT, file), dest);
  }
}
console.error(`${flag === '--clean' ? 'removed' : 'staged'} ${FILES.join(', ')} ${flag === '--clean' ? 'from' : 'into'} packages/${name}/`);
