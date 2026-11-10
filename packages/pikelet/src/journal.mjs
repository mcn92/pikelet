// A journal: a private .pikelet chain an agent may write notes to through
// MCP (docs/agent-journal.md). Each note is one record, appended as its own
// immutable layer; the chain is compacted back into one base before it
// reaches the producer depth limit. A journal is unscored by design: no
// calibration on the base, so drift never matters.
//
// On disk a journal is a directory:
//   journal.json        format, name, creation time, and the chain's member
//                       files, base first
//   base-NNNN.pikelet   the current base (NNNN counts compactions)
//   base-NNNN.LLLL.pikelet  its layers, one note each
//   .lock               held while a write or compaction runs
//
// The directory is kept 0700 and its files 0600 whatever the umask: notes
// can hold conversation-derived and internal material.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { CliError } from './common.mjs';

export const JOURNAL_FORMAT = 'pikelet-journal-v1';
export const NOTE_KINDS = Object.freeze(['handoff', 'decision', 'dead-end', 'measurement', 'summary', 'todo', 'preference']);
export const NOTE_LIMITS = Object.freeze({
  titleChars: 200,
  textBytes: 16 * 1024,
  tags: 20,
  tagChars: 40,
  sources: 20,
  repoListItems: 50,
  repoItemChars: 300,
});
// One note per layer, and a chain may be at most this deep (the producer
// limit append enforces, LAYERED_PROFILE.md 6.1): at this depth the journal
// compacts before writing the next note.
const MAX_DEPTH = 8;
const STALE_LOCK_MS = 10 * 60 * 1000;
const LOCK_WAIT_MS = 30 * 1000;
// A breaker holds .lock.break for one read and one unlink.
const BREAK_STALE_MS = 10 * 1000;
const DIR_MODE = 0o700;
const FILE_MODE = 0o600;

const quietLog = () => {};

export function defaultJournalRoot() {
  return path.join(os.homedir(), '.pikelet', 'journals');
}

// A bare name lives under the default root; anything that looks like a path
// is taken as one.
export function resolveJournalDir(nameOrPath) {
  if (typeof nameOrPath !== 'string' || !nameOrPath.trim()) {
    throw new CliError('a journal is named by a name or a directory path');
  }
  const v = nameOrPath.trim();
  if (v.includes('/') || v.includes('\\') || v.startsWith('.') || path.isAbsolute(v)) return path.resolve(v);
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(v)) {
    throw new CliError(`journal name ${JSON.stringify(v)} must be letters, digits, '.', '_' or '-'`);
  }
  return path.join(defaultJournalRoot(), v);
}

function gitRepoContaining(dir) {
  let cur = path.resolve(dir);
  for (;;) {
    if (fssync.existsSync(path.join(cur, '.git'))) return cur;
    const up = path.dirname(cur);
    if (up === cur) return null;
    cur = up;
  }
}

async function readState(dir) {
  let text;
  try {
    text = await fs.readFile(path.join(dir, 'journal.json'), 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') throw new CliError(`${dir} is not a journal (no journal.json); create one with \`pikelet journal init\``);
    throw err;
  }
  const state = JSON.parse(text);
  if (state.format !== JOURNAL_FORMAT || !Array.isArray(state.members) || !state.members.length) {
    throw new CliError(`${dir}/journal.json is not a ${JOURNAL_FORMAT} journal`);
  }
  for (const m of state.members) {
    if (typeof m !== 'string' || m.includes('/') || m.includes('\\')) {
      throw new CliError(`${dir}/journal.json lists a member outside the journal directory: ${JSON.stringify(m)}`);
    }
  }
  return state;
}

async function writeState(dir, state) {
  const tmp = path.join(dir, `journal.json.${process.pid}.tmp`);
  await fs.writeFile(tmp, `${JSON.stringify(state, null, 2)}\n`, { encoding: 'utf8', mode: FILE_MODE });
  await fs.rename(tmp, path.join(dir, 'journal.json'));
}

/** The chain's member files, absolute, base first. */
export async function journalMembers(dir) {
  const state = await readState(dir);
  return state.members.map((m) => path.join(dir, m));
}

// One writer at a time: two local sessions appending to the same journal
// would otherwise fork the chain. A writer that finds the lock held by a live
// process waits for it, up to LOCK_WAIT_MS: agents finishing a session at the
// same moment each get their note in rather than a refusal. A lock left by a
// dead process, or older than STALE_LOCK_MS, is broken.
//
// The holder record is written whole under a private name and hard-linked
// into place: link fails with EEXIST when .lock exists, and a contender never
// reads a lock that exists but is still empty (an empty lock used to parse as
// stale and be taken over from a live writer).
async function withLock(dir, fn, { waitMs = LOCK_WAIT_MS } = {}) {
  const lockPath = path.join(dir, '.lock');
  const token = crypto.randomBytes(8).toString('hex');
  const body = JSON.stringify({ pid: process.pid, at: Date.now(), token });
  const mine = path.join(dir, `.lock.${process.pid}.${token}`);
  const deadline = Date.now() + waitMs;
  let pause = 20;
  await fs.writeFile(mine, body, { mode: FILE_MODE });
  try {
    for (;;) {
      try {
        await fs.link(mine, lockPath);
        break;
      } catch (err) {
        if (err.code !== 'EEXIST') throw err;
      }
      const raw = await readLockFile(lockPath);
      // Released between our link and our read: try again.
      if (raw === null) continue;
      const holder = parseLock(raw);
      if (isStaleLock(holder, STALE_LOCK_MS)) {
        // Broken under .lock.break, so this cannot remove a lock some other
        // contender took after breaking the same stale one.
        if (await breakStaleLock(dir, lockPath, raw, token)) continue;
      }
      if (Date.now() >= deadline) {
        throw new CliError(`journal ${dir} is locked by process ${holder?.pid ?? '?'}; another session has been writing to it for over ${Math.round(waitMs / 1000)}s`);
      }
      // Jittered backoff so waiting writers do not retry in lockstep.
      await new Promise((r) => setTimeout(r, Math.min(pause * (0.5 + Math.random()), Math.max(0, deadline - Date.now()))));
      pause = Math.min(pause * 2, 250);
    }
  } finally {
    await fs.rm(mine, { force: true });
  }
  try {
    return await fn();
  } finally {
    // Release only our own lock: one broken as stale is not ours to remove.
    if (await readLockFile(lockPath).catch(() => null) === body) await fs.rm(lockPath, { force: true });
  }
}

// A lock's contents, or null when there is none. Any other failure (.lock is
// a directory, or unreadable) is thrown: retrying it would spin until killed.
async function readLockFile(lockPath) {
  try {
    return await fs.readFile(lockPath, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw new CliError(`cannot read the journal lock ${lockPath}: ${err.message}`);
  }
}

function parseLock(raw) {
  try { return JSON.parse(raw); } catch { return null; }
}

// Unparseable, held by a process that no longer exists, or older than maxAgeMs.
function isStaleLock(holder, maxAgeMs) {
  if (!holder || !Number.isSafeInteger(holder.pid)) return true;
  let alive;
  try { process.kill(holder.pid, 0); alive = true; } catch (e) { alive = e.code === 'EPERM'; }
  return !alive || Date.now() - (holder.at || 0) > maxAgeMs;
}

// Remove the stale lock `raw`, if .lock still holds it. Breakers take
// .lock.break first, so between the check and the removal no other breaker
// can free .lock for a new holder; a dead holder cannot release it either
// (a live one past STALE_LOCK_MS could, which is why that age is far beyond
// any write).
// Returns false when another breaker is at work (the caller waits and looks
// again). A .lock.break left by a crashed breaker is itself broken; two
// breakers judging it stale at once is the one window left, and it needs a
// breaker to have died mid-break.
async function breakStaleLock(dir, lockPath, raw, token) {
  const breakPath = path.join(dir, '.lock.break');
  const breakBody = JSON.stringify({ pid: process.pid, at: Date.now(), token });
  const mine = path.join(dir, `.lock.break.${process.pid}.${token}`);
  await fs.writeFile(mine, breakBody, { mode: FILE_MODE });
  try {
    try {
      await fs.link(mine, breakPath);
    } catch (err) {
      if (err.code !== 'EEXIST') throw err;
      const other = await readLockFile(breakPath);
      if (other !== null && isStaleLock(parseLock(other), BREAK_STALE_MS)) await fs.rm(breakPath, { force: true });
      return false;
    }
  } finally {
    await fs.rm(mine, { force: true });
  }
  try {
    if (await readLockFile(lockPath) === raw) await fs.rm(lockPath, { force: true });
    return true;
  } finally {
    if (await readLockFile(breakPath).catch(() => null) === breakBody) await fs.rm(breakPath, { force: true });
  }
}

// A chain file: a base or one of its layers.
const CHAIN_FILE = /^base-\d{4}(\.\d{4})?\.pikelet$/;
// Files a journal directory can hold without journal.json: chain files, a
// journal.json write, and lock files, all left by an interrupted init or
// write.
const JOURNAL_LEFTOVER = /^(base-\d{4}(\.\d{4})?\.pikelet|journal\.json\.\d+\.tmp|\.lock(\.break)?(\.\d+\.[0-9a-f]+)?)$/;

// Tighten the journal directory to 0700 and its chain files and journal.json
// to 0600. Run before a write, so a journal made before modes were enforced
// is closed off before anything new lands in it, and after, because compile,
// append and compact create their outputs under the umask. Filesystems
// without Unix modes (FAT, some mounted Windows drives) accept the chmod and
// keep the old mode; that is reported once per directory, not fatal.
// Windows has no such modes.
const warnedOpen = new Set();
async function makePrivate(dir, log) {
  if (process.platform === 'win32') return;
  await fs.chmod(dir, DIR_MODE);
  for (const f of await fs.readdir(dir)) {
    if (f === 'journal.json' || CHAIN_FILE.test(f)) await fs.chmod(path.join(dir, f), FILE_MODE);
  }
  const mode = (await fs.stat(dir)).mode & 0o777;
  if ((mode & 0o077) && !warnedOpen.has(dir)) {
    warnedOpen.add(dir);
    log(`warning: ${dir} stays mode ${mode.toString(8)} after chmod ${DIR_MODE.toString(8)}; this filesystem does not enforce Unix permissions, so other local users may be able to read the journal`);
  }
}

/**
 * Create an empty journal: a base holding one seed record (append needs a
 * base with at least one record), compiled without calibration.
 */
export async function initJournal({ nameOrPath, force = false, log = (l) => console.log(l) }) {
  const dir = resolveJournalDir(nameOrPath);
  const statePath = path.join(dir, 'journal.json');
  // A journal.json that is not a Pikelet journal's (a project's own config,
  // say) makes the directory foreign: --force replaces journals, nothing else.
  let existing = false;
  if (fssync.existsSync(statePath)) {
    try {
      await readState(dir);
      existing = true;
    } catch {
      throw new CliError(`${dir} holds a journal.json that is not a ${JOURNAL_FORMAT} journal; journal init will not replace it, even with --force. `
        + 'Name a new directory, or a bare name for one under ~/.pikelet/journals');
    }
  }
  if (existing && !force) throw new CliError(`${dir} already holds a journal; pass --force to replace it`);
  // init makes its directory 0700, so it only takes a directory that is, or
  // is becoming, a journal: a new or empty one, an existing journal, or one
  // holding only what an interrupted init or write leaves behind. Pointed at
  // a shared folder by mistake, it refuses before changing anything.
  if (!existing) {
    let entries = [];
    try { entries = await fs.readdir(dir); } catch (err) { if (err.code !== 'ENOENT') throw err; }
    const foreign = entries.filter((f) => !JOURNAL_LEFTOVER.test(f));
    if (foreign.length) {
      throw new CliError(`${dir} is not empty and is not a journal (it holds ${foreign.slice(0, 3).map((f) => JSON.stringify(f)).join(', ')}${foreign.length > 3 ? ', ...' : ''}); `
        + 'journal init makes its directory private (0700), so it only creates a journal in a new or empty directory. '
        + 'Name a new directory, or a bare name for one under ~/.pikelet/journals');
    }
  }
  const repo = gitRepoContaining(dir);
  if (repo) {
    log(`warning: ${dir} is inside the git repository ${repo}. A journal is private by default;`);
    log(`  keep it out of version control, e.g. add "${path.relative(repo, dir) || '.'}/" to ${path.join(repo, '.gitignore')}`);
  }
  await fs.mkdir(dir, { recursive: true, mode: DIR_MODE });
  await makePrivate(dir, log);
  // Replacing a journal holds the lock, so it cannot interleave with a
  // write, and removes the old chain files: a layer left on disk would
  // otherwise collide with the new journal's first append.
  if (existing) {
    return withLock(dir, async () => {
      for (const f of await fs.readdir(dir)) {
        if (CHAIN_FILE.test(f)) await fs.rm(path.join(dir, f), { force: true });
      }
      return createJournalFiles(dir, log);
    });
  }
  return createJournalFiles(dir, log);
}

async function createJournalFiles(dir, log) {
  const name = path.basename(dir);
  const seedDir = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-journal-seed-'));
  try {
    await fs.writeFile(path.join(seedDir, 'journal.md'), `# Journal: ${name}\n\n`
      + 'This is a private Pikelet journal. Its other records are notes that coding agents wrote through '
      + 'the pikelet MCP server while working: handoffs, decisions, dead ends, measurements and summaries. '
      + 'Every note is agent-written and unverified; read it as a colleague\'s working notes, not as a '
      + 'published source.\n', 'utf8');
    const { compileArtifact } = await import('./cli.mjs');
    const baseFile = 'base-0001.pikelet';
    await compileArtifact({ source: seedDir, out: path.join(dir, baseFile), 'skip-calibration': true, force: true });
    await writeState(dir, {
      format: JOURNAL_FORMAT,
      name,
      created: new Date().toISOString(),
      generation: 1,
      members: [baseFile],
    });
    await makePrivate(dir, log);
    log(`Created journal ${name} at ${dir}`);
    return { dir, name };
  } finally {
    await fs.rm(seedDir, { recursive: true, force: true });
  }
}

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const boundedString = (v, max, what) => {
  if (typeof v !== 'string' || !v.trim()) throw new Error(`${what} must be a non-empty string`);
  if (v.length > max) throw new Error(`${what} is longer than ${max} characters`);
  return v.trim();
};

/**
 * Check a note as an agent submitted it and shape it into a record. The
 * server-side fields (created, authoredBy, session) are passed separately
 * and are never taken from the agent's arguments.
 */
export function shapeNote(args, { created, authoredBy, session }) {
  if (!isPlainObject(args)) throw new Error('write_note takes an object');
  const kind = args.kind;
  if (!NOTE_KINDS.includes(kind)) throw new Error(`kind must be one of ${NOTE_KINDS.join(', ')}`);
  const title = boundedString(args.title, NOTE_LIMITS.titleChars, 'title');
  if (title.includes('\n')) throw new Error('title must be one line');
  if (typeof args.text !== 'string' || !args.text.trim()) throw new Error('text must be a non-empty string');
  if (Buffer.byteLength(args.text, 'utf8') > NOTE_LIMITS.textBytes) {
    throw new Error(`text is larger than ${NOTE_LIMITS.textBytes} bytes; split it into several notes`);
  }
  const text = args.text.trim();
  let tags = [];
  if (args.tags !== undefined) {
    if (!Array.isArray(args.tags) || args.tags.length > NOTE_LIMITS.tags) throw new Error(`tags must be an array of at most ${NOTE_LIMITS.tags} strings`);
    tags = [...new Set(args.tags.map((t, i) => boundedString(t, NOTE_LIMITS.tagChars, `tags[${i}]`).toLowerCase()))];
  }
  let repo;
  if (args.repo !== undefined) {
    if (!isPlainObject(args.repo)) throw new Error('repo must be an object');
    repo = {};
    for (const key of ['repository', 'branch']) {
      if (args.repo[key] !== undefined) repo[key] = boundedString(args.repo[key], NOTE_LIMITS.repoItemChars, `repo.${key}`);
    }
    for (const key of ['commits', 'files', 'versions']) {
      if (args.repo[key] === undefined) continue;
      if (!Array.isArray(args.repo[key]) || args.repo[key].length > NOTE_LIMITS.repoListItems) {
        throw new Error(`repo.${key} must be an array of at most ${NOTE_LIMITS.repoListItems} strings`);
      }
      repo[key] = args.repo[key].map((v, i) => boundedString(v, NOTE_LIMITS.repoItemChars, `repo.${key}[${i}]`));
    }
    const unknown = Object.keys(args.repo).filter((k) => !['repository', 'branch', 'commits', 'files', 'versions'].includes(k));
    if (unknown.length) throw new Error(`repo does not take ${unknown.join(', ')}`);
  }
  let sources;
  if (args.sources !== undefined) {
    if (!Array.isArray(args.sources) || args.sources.length > NOTE_LIMITS.sources) {
      throw new Error(`sources must be an array of at most ${NOTE_LIMITS.sources} citations`);
    }
    sources = args.sources.map((s, i) => {
      if (!isPlainObject(s) || !/^[0-9a-f]{64}$/i.test(String(s.identity)) || !Number.isSafeInteger(s.id) || s.id < 0) {
        throw new Error(`sources[${i}] must be { identity: <pack sha256>, id: <record id> }`);
      }
      return { identity: String(s.identity).toLowerCase(), id: s.id, ...(typeof s.pack === 'string' ? { pack: s.pack.slice(0, 100) } : {}) };
    });
  }
  const unknown = Object.keys(args).filter((k) => !['kind', 'title', 'text', 'tags', 'repo', 'sources'].includes(k));
  if (unknown.length) throw new Error(`write_note does not take ${unknown.join(', ')}`);
  const note = {
    kind, created, authoredBy, session, tags,
    ...(repo && Object.keys(repo).length ? { repo } : {}),
    ...(sources?.length ? { sources } : {}),
  };
  // The title is the record's first line, as a document chunk's heading is,
  // so search and grounding see it with the body.
  const id = `note-${created}-${crypto.randomBytes(3).toString('hex')}`;
  return {
    id, docId: id, title, headingPath: [kind], url: null, anchor: null,
    sourcePath: `journal/${kind}/${created}`,
    text: `${title}\n${text}`,
    note,
  };
}

async function compactLocked(dir, state, log) {
  const { compactChain } = await import('./compact.mjs');
  const generation = (state.generation || 1) + 1;
  const baseFile = `base-${String(generation).padStart(4, '0')}.pikelet`;
  const out = path.join(dir, baseFile);
  const written = await compactChain({ head: state.members.map((m) => path.join(dir, m)), out, 'no-refit': true, force: true }, { log });
  const old = state.members;
  const next = { ...state, generation, members: [baseFile], lastCompacted: new Date().toISOString() };
  await writeState(dir, next);
  for (const m of old) await fs.rm(path.join(dir, m), { force: true });
  return { state: next, written };
}

/** Fold the journal's layers back into one base. */
export async function compactJournal(nameOrPath, { log = (l) => console.log(l), lockWaitMs } = {}) {
  const dir = resolveJournalDir(nameOrPath);
  // Confirm this is a journal before locking or tightening anything: run on
  // the wrong directory, it must fail without writing a lock file into it or
  // changing its mode.
  await readState(dir);
  return withLock(dir, async () => {
    const state = await readState(dir);
    await makePrivate(dir, log);
    if (state.members.length === 1) {
      log('The journal is already a single base; nothing to compact');
      return { dir, members: state.members };
    }
    const { state: next } = await compactLocked(dir, state, log);
    await makePrivate(dir, log);
    return { dir, members: next.members };
  }, { waitMs: lockWaitMs });
}

/**
 * Append one shaped note (from shapeNote) as a new layer, compacting first
 * when the chain is at the depth limit. Returns the note's chain-global id
 * and the new head's identity.
 */
export async function appendNote(nameOrPath, record, { log = quietLog, lockWaitMs } = {}) {
  const dir = resolveJournalDir(nameOrPath);
  // As in compactJournal: not a journal means no lock file and no chmod.
  await readState(dir);
  return withLock(dir, async () => {
    let state = await readState(dir);
    await makePrivate(dir, log);
    if (state.members.length - 1 >= MAX_DEPTH) {
      log(`Journal at depth ${state.members.length - 1}: compacting before the next note`);
      ({ state } = await compactLocked(dir, state, log));
    }
    const base = state.members[0].replace(/\.pikelet$/, '');
    const layerFile = `${base}.${String(state.members.length).padStart(4, '0')}.pikelet`;
    // A file by this name is not in the chain (members.length indexes past
    // its last layer): a write that died between the layer and journal.json.
    // Holding the lock, it is ours to clear; left alone it blocks every
    // later append with "Output file already exists".
    if (fssync.existsSync(path.join(dir, layerFile))) {
      log(`Removing ${layerFile}: left by an interrupted write, not part of the chain`);
      await fs.rm(path.join(dir, layerFile), { force: true });
    }
    const { appendLayer } = await import('./append.mjs');
    const written = await appendLayer({
      parent: state.members.map((m) => path.join(dir, m)),
      out: path.join(dir, layerFile),
      // Unscored by design: drift has no fit to outgrow.
      'allow-drift': true,
    }, { chunks: [record], log });
    await writeState(dir, { ...state, members: [...state.members, layerFile] });
    await makePrivate(dir, log);
    return { id: written.rowBase, identity: written.identity, dir };
  }, { waitMs: lockWaitMs });
}

/** Notes newest first, read from an open chain (layer-reader's openPikeletChain). */
export async function latestNotes(chain, { kind, tag, limit = 10 } = {}) {
  const info = chain.info();
  const out = [];
  for (let id = info.records - 1; id >= 0 && out.length < limit; id--) {
    const r = await chain.record(id);
    if (!r || r.tombstoned || !r.note) continue;
    if (kind && r.note.kind !== kind) continue;
    if (tag && !(r.note.tags || []).includes(String(tag).toLowerCase())) continue;
    out.push(r);
  }
  return out;
}

export async function openJournalChain(nameOrPath) {
  const dir = resolveJournalDir(nameOrPath);
  const members = await journalMembers(dir);
  const { openPikeletChain } = await import('pikelet-wasm/complete/layer-reader.mjs');
  const chain = await openPikeletChain(members);
  return { dir, chain };
}
