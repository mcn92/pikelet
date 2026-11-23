// Git history as a source: one document per commit, so a pack can answer
// "why is it like this?" and "did we already try that?" from the commit
// messages, citing the commit that says so.
//
// git runs with an argument vector (never a shell) and with the repository's
// own config neutralized where it could run a program: a cloned repository
// is not trusted to execute anything just because it is being read.

import { execFile } from 'node:child_process';
import fssync from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import { CliError } from './common.mjs';
import { trimCharEnd } from './text-scan.mjs';

const execFileAsync = promisify(execFile);

// Separators git cannot put inside a field it prints: %x1e opens a commit,
// %x1f ends each field. The last field, after the final %x1f, is the
// --name-only file list.
const LOG_FORMAT = '%x1e%H%x1f%aI%x1f%an%x1f%s%x1f%b%x1f';
// A commit that touched hundreds of files (a rename sweep, a vendored
// dependency) would otherwise turn its record into a file list.
const MAX_FILES_LISTED = 40;
const MAX_LOG_BYTES = 512 * 1024 * 1024;

async function git(repo, args) {
  try {
    const { stdout } = await execFileAsync('git', [
      '-C', repo,
      // Nothing here needs a pager, a terminal prompt, an fsmonitor hook,
      // or signature verification; each is a way repository config could
      // run a program or block.
      '-c', 'core.fsmonitor=false',
      '-c', 'core.pager=cat',
      '-c', 'log.showSignature=false',
      '-c', 'core.quotepath=off',
      ...args,
    ], {
      maxBuffer: MAX_LOG_BYTES,
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_PAGER: 'cat' },
    });
    return stdout;
  } catch (error) {
    if (error.code === 'ENOENT') {
      throw new CliError('git source needs the git executable on PATH', 1);
    }
    const detail = String(error.stderr || error.message).trim().split('\n')[0];
    throw new CliError(`git ${args[0]} failed in ${repo}: ${detail}`, 1);
  }
}

// https://github.com/<owner>/<repo>/commit/<sha> when origin is on GitHub,
// else null and records cite the commit by its sha alone. Only host and path
// are kept: an origin URL can carry credentials (https://<token>@github.com/...),
// and those must never reach a pack.
function commitUrlBase(remote) {
  const trimmed = String(remote || '').trim();
  let host;
  let repoPath;
  const scp = /^[^@\s/]+@([^:\s/]+):(.+)$/.exec(trimmed);
  if (scp) {
    [, host, repoPath] = scp;
  } else {
    try {
      const url = new URL(trimmed);
      if (!['https:', 'http:', 'ssh:', 'git:'].includes(url.protocol)) return null;
      host = url.hostname;
      repoPath = url.pathname;
    } catch {
      return null;
    }
  }
  if (host.toLowerCase() !== 'github.com') return null;
  const parts = trimCharEnd(repoPath.replace(/^\/+/, '').replace(/\.git$/, ''), '/').split('/');
  if (parts.length !== 2 || !parts.every((p) => /^[A-Za-z0-9._-]+$/.test(p))) return null;
  return `https://github.com/${parts[0]}/${parts[1]}/commit/`;
}

function parseLog(stdout) {
  const commits = [];
  for (const entry of stdout.split('\x1e')) {
    if (!entry.trim()) continue;
    const fields = entry.split('\x1f');
    if (fields.length < 6) continue;
    // A separator byte inside a message body would split it further; the
    // fixed fields are at the ends, so rejoin whatever lies between.
    const [sha, date, author, subject] = fields;
    const fileBlock = fields[fields.length - 1];
    const body = fields.slice(4, -1).join(' ');
    commits.push({
      sha: sha.trim(),
      date: date.trim(),
      author: author.trim(),
      subject: subject.trim(),
      body: body.trim(),
      files: fileBlock.split('\n').map((f) => f.trim()).filter(Boolean),
    });
  }
  return commits;
}

function commitText(commit) {
  const files = commit.files.length > MAX_FILES_LISTED
    ? `${commit.files.slice(0, MAX_FILES_LISTED).join(', ')} and ${commit.files.length - MAX_FILES_LISTED} more`
    : commit.files.join(', ');
  return [
    commit.subject,
    commit.body,
    `Commit ${commit.sha.slice(0, 12)} by ${commit.author} on ${commit.date.slice(0, 10)}.`,
    files ? `Files changed: ${files}.` : '',
  ].filter(Boolean).join('\n\n');
}

// The repository's name for a default pack name: the GitHub repo when origin
// is on GitHub (a checkout's folder can be named anything), else the folder.
async function gitRepoName(repoPath) {
  const top = (await git(path.resolve(repoPath), ['rev-parse', '--show-toplevel'])).trim();
  let remote = '';
  try { remote = await git(top, ['remote', 'get-url', 'origin']); } catch { /* no origin */ }
  const base = commitUrlBase(remote);
  return base ? base.split('/')[4] : path.basename(top);
}

async function ingestGit(source, log = () => {}) {
  const repo = path.resolve(source.path);
  if (!fssync.existsSync(repo)) {
    throw new CliError(`Git repository not found: ${repo}\nNext: check --git`, 1);
  }
  const rev = source.rev || 'HEAD';
  // A rev that looks like a flag would be read by git as one.
  if (rev.startsWith('-')) throw new CliError(`--git-rev must be a revision or range, got ${rev}`, 1);
  const top = (await git(repo, ['rev-parse', '--show-toplevel'])).trim();
  let remote = '';
  try { remote = await git(top, ['remote', 'get-url', 'origin']); } catch { /* no origin: cite by sha */ }
  const urlBase = commitUrlBase(remote);
  const stdout = await git(top, ['log', '--no-color', '--no-merges', '--name-only', `--format=${LOG_FORMAT}`, rev, '--']);
  const commits = parseLog(stdout);
  log(`Read ${commits.length} commit(s) from ${top} (${rev}, merges excluded)`
    + (urlBase ? `; citing ${urlBase.replace(/\/commit\/$/, '')}` : '; no GitHub origin, citing by sha'));
  // Oldest first, so record ids grow with history and a later append of new
  // commits continues the order.
  return commits.reverse().map((commit, id) => ({
    id,
    sourcePath: `commit/${commit.sha}`,
    url: urlBase ? `${urlBase}${commit.sha}` : null,
    title: commit.subject || commit.sha.slice(0, 12),
    slug: null,
    text: commitText(commit),
    sections: [],
    commit: { sha: commit.sha, date: commit.date, author: commit.author },
  }));
}

export { ingestGit, gitRepoName, commitUrlBase, parseLog, commitText };
