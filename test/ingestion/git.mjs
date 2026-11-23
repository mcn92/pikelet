// ingestGit conformance: a repository's commit history becomes one document
// per commit, oldest first, carrying its message, date, author and files,
// cited by GitHub URL only when origin is on GitHub, and never with the
// credentials an origin URL can carry.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, '..', '..', 'packages', 'pikelet', 'src');
const { ingestGit, gitRepoName, commitUrlBase, parseLog } = await import(pathToFileURL(path.join(src, 'git-source.mjs')).href);
const { chunkDocs, publicChunk } = await import(pathToFileURL(path.join(src, 'ingest.mjs')).href);
const { compileArtifact } = await import(pathToFileURL(path.join(src, 'cli.mjs')).href);

const env = {
  ...process.env,
  GIT_AUTHOR_NAME: 'Ada Tester', GIT_AUTHOR_EMAIL: 'ada@example.com',
  GIT_COMMITTER_NAME: 'Ada Tester', GIT_COMMITTER_EMAIL: 'ada@example.com',
  GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: os.devNull,
};
const git = (cwd, ...args) => execFileSync('git', args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] }).toString();
const commit = async (repo, file, content, message) => {
  await fs.mkdir(path.dirname(path.join(repo, file)), { recursive: true });
  await fs.writeFile(path.join(repo, file), content);
  git(repo, 'add', '-A');
  git(repo, 'commit', '-q', '-m', message);
};

const rejects = async (fn, pattern) => {
  try { await fn(); } catch (error) { return pattern.test(error.message) ? true : error.message; }
  return 'did not throw';
};

section('commitUrlBase: GitHub origins only, credentials dropped');
{
  check('an https origin with a token loses the token',
    commitUrlBase('https://ghp_secret:x-oauth@github.com/acme/widgets.git') === 'https://github.com/acme/widgets/commit/');
  check('an scp-style ssh origin maps to the https commit URL',
    commitUrlBase('git@github.com:acme/widgets.git') === 'https://github.com/acme/widgets/commit/');
  check('an ssh:// origin maps to the https commit URL',
    commitUrlBase('ssh://git@github.com/acme/widgets') === 'https://github.com/acme/widgets/commit/');
  check('a non-GitHub host gets no URL', commitUrlBase('https://gitlab.com/acme/widgets.git') === null);
  check('a path that is not owner/repo gets no URL', commitUrlBase('https://github.com/acme/widgets/extra') === null);
  check('an encoded path segment gets no URL', commitUrlBase('https://github.com/acme/w%2f..') === null);
  check('a file:// origin gets no URL', commitUrlBase('file:///srv/widgets') === null);
  check('no origin gets no URL', commitUrlBase('') === null);
}

section('parseLog: a separator byte inside a body does not shift the fields');
{
  const out = parseLog(`\x1eabc\x1f2026-01-02T03:04:05+00:00\x1fAda\x1fSubject\x1fbody one\x1fbody two\x1f\n\nsrc/a.js\nsrc/b.js\n`);
  check('one commit parsed', out.length === 1, JSON.stringify(out));
  check('the files are still the files', JSON.stringify(out[0]?.files) === '["src/a.js","src/b.js"]', JSON.stringify(out[0]?.files));
  check('the body keeps both halves', out[0]?.body === 'body one body two', JSON.stringify(out[0]?.body));
}

section('ingestGit: one document per commit, oldest first');
{
  const repo = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-ingest-git-'));
  try {
    git(repo, 'init', '-q', '-b', 'main');
    await commit(repo, 'README.md', 'one\n', 'Add a README\n\nWhy: the project needs a front page that explains what it does and how to install it.');
    await commit(repo, 'src/sync.js', 'export {}\n', 'Retry uploads on 503\n\nWhat changed: uploads retry three times with backoff.\nWhy: the storage API sheds load with 503s.');
    git(repo, 'checkout', '-q', '-b', 'side');
    await commit(repo, 'docs/side.md', 'side\n', 'Document the side branch work in enough words to be kept as a record');
    git(repo, 'checkout', '-q', 'main');
    git(repo, 'merge', '-q', '--no-ff', '-m', 'Merge branch side', 'side');
    const many = Array.from({ length: 45 }, (_, i) => `gen/f${String(i).padStart(2, '0')}.txt`);
    for (const f of many) {
      await fs.mkdir(path.join(repo, 'gen'), { recursive: true });
      await fs.writeFile(path.join(repo, f), f);
    }
    git(repo, 'add', '-A');
    git(repo, 'commit', '-q', '-m', 'Generate fixtures for the sync tests across every supported platform');

    const logs = [];
    const docs = await ingestGit({ path: repo }, (l) => logs.push(l));
    check('every non-merge commit is a document', docs.length === 4, JSON.stringify(docs.map((d) => d.title)));
    check('the merge commit is excluded', !docs.some((d) => d.title.startsWith('Merge branch')));
    check('oldest first, with ids in order', docs[0]?.title === 'Add a README' && docs.every((d, i) => d.id === i),
      JSON.stringify(docs.map((d) => [d.id, d.title])));
    const retry = docs.find((d) => d.title === 'Retry uploads on 503');
    check('the subject is the title and the body is in the text',
      retry && retry.text.startsWith('Retry uploads on 503') && retry.text.includes('the storage API sheds load'), retry?.text);
    check('the text names the short sha, author and date',
      retry && retry.text.includes(`Commit ${retry.commit.sha.slice(0, 12)} by Ada Tester on ${retry.commit.date.slice(0, 10)}.`), retry?.text);
    check('the text lists the files changed', retry?.text.includes('Files changed: src/sync.js.'), retry?.text);
    check('the record carries the full sha, date and author',
      /^[0-9a-f]{40}$/.test(retry?.commit.sha) && retry.commit.author === 'Ada Tester' && !Number.isNaN(Date.parse(retry.commit.date)),
      JSON.stringify(retry?.commit));
    check('sourcePath names the commit', retry?.sourcePath === `commit/${retry?.commit.sha}`, retry?.sourcePath);
    check('no origin: no URL, and the log says so', retry?.url === null && logs.some((l) => l.includes('citing by sha')), JSON.stringify(logs));
    check('the author email never reaches the text', !docs.some((d) => d.text.includes('ada@example.com')));
    const gen = docs.find((d) => d.title.startsWith('Generate fixtures'));
    check('a long file list is cut at 40 with a count', gen?.text.includes('and 5 more.') && !gen.text.includes('gen/f44.txt'), gen?.text.slice(-120));

    const chunks = chunkDocs(docs, {});
    check('chunks keep the commit', chunks.length > 0 && chunks.every((c) => c.commit?.sha), JSON.stringify(chunks.map((c) => c.commit)));
    check('the public record keeps the commit', publicChunk(chunks[0]).commit?.sha === chunks[0].commit.sha);

    const firstSha = git(repo, 'rev-list', '--max-parents=0', 'HEAD').trim();
    const ranged = await ingestGit({ path: repo, rev: `${firstSha}..main` });
    check('a revision range limits the commits', ranged.length === 3 && !ranged.some((d) => d.title === 'Add a README'),
      JSON.stringify(ranged.map((d) => d.title)));
    check('a rev that looks like a flag is refused',
      await rejects(() => ingestGit({ path: repo, rev: '--output=/tmp/x' }), /must be a revision/) === true);

    git(repo, 'remote', 'add', 'origin', 'https://ghp_secret@github.com/acme/widgets.git');
    const linkedLogs = [];
    const linked = await ingestGit({ path: repo }, (l) => linkedLogs.push(l));
    check('a GitHub origin gives each record its commit URL',
      linked.every((d) => d.url === `https://github.com/acme/widgets/commit/${d.commit.sha}`), linked[0]?.url);
    check('the origin token appears nowhere', !JSON.stringify(linked).includes('ghp_secret') && !linkedLogs.some((l) => l.includes('ghp_secret')),
      JSON.stringify(linkedLogs));
    check('the log names the GitHub repo it cites', linkedLogs.some((l) => l.includes('citing https://github.com/acme/widgets')), JSON.stringify(linkedLogs));
    check('the default name comes from the GitHub repo', await gitRepoName(repo) === 'widgets');
    git(repo, 'remote', 'remove', 'origin');
    check('without origin the default name is the folder', await gitRepoName(repo) === path.basename(repo));
  } finally {
    await fs.rm(repo, { recursive: true, force: true });
  }
}

section('ingestGit: errors');
{
  const plain = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-ingest-nogit-'));
  try {
    check('a folder that is not a repository is refused with git\'s reason',
      await rejects(() => ingestGit({ path: plain }), /git rev-parse failed/) === true);
    check('a missing path is refused', await rejects(() => ingestGit({ path: path.join(plain, 'nope') }), /not found/) === true);
  } finally {
    await fs.rm(plain, { recursive: true, force: true });
  }
}

section('compile --git: flag combinations');
{
  const out = path.join(os.tmpdir(), `pikelet-git-flags-${process.pid}.pikelet`);
  check('--git with --source is refused',
    await rejects(() => compileArtifact({ git: '.', source: './docs', out }), /one of --source, --git or --conversations, got --source and --git/) === true);
  check('--git-rev without --git is refused',
    await rejects(() => compileArtifact({ source: './docs', 'git-rev': 'HEAD', out }), /--git-rev requires --git/) === true);
  check('folder filters with --git are refused',
    await rejects(() => compileArtifact({ git: '.', include: ['*.md'], out }), /filter --source/) === true);
  check('neither --source nor --git is refused',
    await rejects(() => compileArtifact({ out }), /requires --source <path\|url>, --git <repo> or --conversations/) === true);
}
