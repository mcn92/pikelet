// pikelet CLI: argument parsing, create / compile / rebuild /
// doctor commands, and the config the scaffold is generated from. The work
// happens in the sibling modules (see the module map in README / each file
// header).

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { CONFIG_SCHEMA_URL, DEFAULT_CONFIG, MODEL_MAP, CliError, modelCacheDir } from './common.mjs';
import { writeProject, writeRuntimeModules, stageStudentInputs, wranglerToml, deployProject } from './scaffold.mjs';
import { buildAssets, validateConfig } from './build.mjs';
import { gitRepoName } from './git-source.mjs';

export async function main(argv) {
  const parsed = parseArgs(argv.slice(2));
  // Help is answered before any command runs, so `mcp --help` prints help
  // instead of complaining that no pack was given. A bare `pikelet` prints
  // the overview: the old interactive scaffold is `pikelet create`.
  const wantsHelp = parsed.flags.help || parsed.flags.h || parsed.positionals.includes('-h');
  if (parsed.positionals[0] === 'help' || wantsHelp) {
    const named = parsed.positionals[0] === 'help' ? parsed.positionals[1] : parsed.positionals[0];
    printHelp(named);
    return;
  }
  if (parsed.positionals.length === 0 && Object.keys(parsed.flags).length === 0) {
    printHelp();
    return;
  }
  if (parsed.positionals[0] === 'doctor') {
    const { runDoctor } = await import('./doctor.mjs');
    const report = await runDoctor(parsed.positionals[1]);
    if (report.failures) {
      throw Object.assign(new Error('doctor found failing hosting checks'), { exitCode: 1 });
    }
    return;
  }
  if (parsed.positionals[0] === 'append') {
    const { appendLayer } = await import('./append.mjs');
    await appendLayer(parsed.flags);
    return;
  }
  if (parsed.positionals[0] === 'compact') {
    const { compactChain } = await import('./compact.mjs');
    await compactChain(parsed.flags);
    return;
  }
  if (parsed.positionals[0] === 'rebase') {
    const { rebaseLayer } = await import('./compact.mjs');
    await rebaseLayer(parsed.flags);
    return;
  }
  if (parsed.positionals[0] === 'journal') {
    const { runJournalCommand } = await import('./journal-cli.mjs');
    await runJournalCommand(parsed.positionals.slice(1), parsed.flags);
    return;
  }
  if (parsed.positionals[0] === 'mcp') {
    const { runMcpServer, loadShelf, installMcpConfig } = await import('./mcp.mjs');
    const { loadCompleteModules, CLI_VERSION, plainHttpWarning } = await import('./common.mjs');
    const warnInsecure = (location) => {
      const warning = plainHttpWarning(location);
      if (warning) process.stderr.write(`${warning}\n`);
    };
    const { nodeFetch } = await import('pikelet-wasm/complete/node-fetch.mjs');
    // Positional pack paths (and URLs, optionally `#sha256`-pinned)
    // compose with repeated --pack; --shelf mounts a static pack listing.
    const positionalPacks = parsed.positionals.slice(1).filter((p) => p !== 'install');
    const packPaths = [...(parsed.flags.pack || []), ...positionalPacks];
    if (parsed.positionals[1] === 'install') {
      const written = await installMcpConfig({
        packPaths,
        shelf: parsed.flags.shelf,
        journal: parsed.flags.journal,
        client: parsed.flags.client || 'claude-code',
        serverName: parsed.flags['server-name'] || 'knowledge-packs',
        force: parsed.flags.force === true,
        version: CLI_VERSION,
      });
      console.log(`Wrote MCP server "${written.serverName}" to ${written.configPath}`);
      console.log(written.hint);
      return;
    }
    if (parsed.flags.shelf) {
      warnInsecure(parsed.flags.shelf);
      packPaths.push(...await loadShelf(parsed.flags.shelf));
    }
    const { reader } = await loadCompleteModules();
    // diskBlobCache comes from the reader's Node entry; a pikelet-wasm that
    // predates it simply mounts uncached.
    const blobCache = parsed.flags['no-cache'] || typeof reader.diskBlobCache !== 'function'
      ? null
      : reader.diskBlobCache();
    await runMcpServer({
      packPaths,
      journal: parsed.flags.journal ?? null,
      openPikeletFile: reader.openPikeletFile,
      // Pack bytes never pass through Node fetch's response decoder.
      httpRangeSource: (url, options = {}) => {
        warnInsecure(url);
        return reader.httpRangeSource(url, { fetch: nodeFetch, ...options });
      },
      serverVersion: CLI_VERSION,
      blobCache,
    });
    return;
  }
  // `pikelet create` is the canonical form; bare flags still scaffold for
  // compatibility with the old `npm create` flow.
  const command = ['rebuild', 'compile', 'create'].includes(parsed.positionals[0]) ? parsed.positionals[0] : 'create';
  if (command === 'rebuild') {
    await rebuildProject(process.cwd(), parsed.flags);
    return;
  }
  if (command === 'compile') {
    await compileArtifact(parsed.flags);
    return;
  }
  await createProject(parsed.flags);
}

// `pikelet --help` is a short overview; `pikelet <command> --help` (or
// `pikelet help <command>`) prints that command's usage and options.
const HELP_OVERVIEW = `pikelet: compile documents into one searchable .pikelet file, and give it
to an AI agent.

Get started:
  pikelet compile --source ./docs --out docs.pikelet
  pikelet mcp install --pack ./docs.pikelet      # add it to Claude Code

Commands:
  compile   Build a .pikelet from a folder, a website, git history or chat exports
  mcp       Serve packs to an MCP client; 'mcp install' writes the client config
  append    Add, remove or replace records as a small layer on top of a pack
  compact   Fold a layered pack back into one file
  rebase    Move a layer onto a different base
  journal   Create and read an agent's private journal
  doctor    Check that a web host serves packs correctly
  create    Scaffold a Cloudflare Worker search project (the older workflow)
  rebuild   Rebuild a scaffolded project's assets

Run 'pikelet <command> --help' for a command's options.`;

const LAYER_TEXT = `append, compact and rebase work on layered chains (LAYERED_PROFILE.md).
append writes a small layer on top of a pack: new records from --source,
tombstones from --remove (one id per flag), and replacements from
--supersede <oldId>=<path>, which retires oldId in favour of the record
ingested from that source path. --parent names the chain base-first, or
just its head, whose ancestors are found by locator. It refuses when the
chain's calibration would drift past its limit unless --allow-drift (the
chain then serves unscored); --max-depth caps the chain depth. compact folds a
chain into one fresh base (--no-refit skips recalibration and ships it
unscored). rebase moves a layer onto a different head, replaying only what
that layer itself did; its original parent is found by the layer's locator
or named with --old-parent. Every chain-member flag (--parent, --head,
--onto, --old-parent, --layer) takes '#<sha256>' to pin that member's
manifest identity. A new layer records where its parent lives, as
--parent-locator <relative path> or, by default, the parent's path relative
to --out when the parent is a local file under the output directory.`;

const HELP = {
  compile: `Usage:
  pikelet compile --source <path|url> --out docs.pikelet
  pikelet compile --git <repo> [--git-rev <rev|range>] --out history.pikelet
  pikelet compile --conversations <file|folder> --out conversations.pikelet

compile builds a complete kind-3 .pikelet artifact from the source and stops:
no project, no Worker, no Cloudflare. The file carries the corpus, index,
inline MiniLM query encoder, calibrated abstention, and evaluation data; open
it with pikelet-wasm/complete on any runtime. Abstention is self-calibrated
from the corpus at build time (skipped with a logged reason when the corpus
cannot support a trustworthy fit); --calibration <file> supplies a prebuilt
retrieval-signals asset instead, --skip-calibration ships the artifact
unscored. --calibration-queries <file> supplies real, corpus-author-written
questions (JSON-lines, one {text, expectId} or {text, expectTitle} object
per line) used only to validate the self-calibrated fit and as the
artifact's embedded golden queries — never as training data for the fit
itself. compile also takes --source, --out, --name, --max-pages, --include,
--exclude, --force (overwrite the output file), and --encoder-model/
--encoder-weights/--encoder-vocab/--encoder-pooling/--encoder-query-prefix/
--encoder-passage-prefix to swap the packaged MiniLM-L6 encoder for another
BERT-shaped one built the same way (see
examples/one-file-search/encoder-spike/export_encoder_blob.py) — the
compiled kernel's dimensions are fixed at build time, so a swapped model
must match MiniLM-L6's shape (384 hidden / 6 layers / 12 heads / 1536 FFN /
30522 vocab) exactly.

compile --git <repo> builds a pack from a repository's commit history instead
of files: one record per commit (subject, message, date, author, files
changed), merges excluded, cited by the commit's GitHub URL when origin is on
GitHub and by its sha otherwise. --git-rev limits it to a revision or range
(default HEAD, e.g. v0.9.0..HEAD); the pack is named <repo>-history unless
--name is given.

compile --conversations <file|folder> builds a pack from past chats: one
record per exchange (a prompt and the reply to it), oldest first. It reads
Claude Code session transcripts (*.jsonl, e.g. a folder under
~/.claude/projects) and the conversations.json in a Claude.ai or ChatGPT data
export, detected by shape; a folder is searched for all of them. Only what
was said is kept: tool calls, tool output and reasoning are dropped, and
common credential shapes (API keys, tokens, private keys, password=...) and
email addresses are redacted. Redaction is pattern-based, a best effort and
not a guarantee: a conversation pack is private by default, so keep it that
way. The pack is named conversations unless --name is given.

Options:
  --source <path|url>   Folder or website to compile
  --out <file>          Output path (default search.pikelet)
  --name <name>         Pack name (default: the folder or host name;
                        <repo>-history for --git, conversations for
                        --conversations)
  --license <spdx>      License of the compiled content, recorded in the
                        pack; packs meant for redistribution should carry one
  --force               Overwrite the output file
  --include <glob>      Folder include glob, repeatable
  --exclude <glob>      Folder exclude glob, repeatable
  --include-url <pat>   URL-path include pattern ('*' wildcard), repeatable
  --exclude-url <pat>   URL-path exclude pattern ('*' wildcard), repeatable;
                        aggregate pages like mdBook's print.html are always
                        excluded
  --max-pages <n>       URL crawl cap
  --git <repo>          Build from a repository's commit history
  --git-rev <rev|range> With --git: revision or range (default HEAD)
  --conversations <file|folder>
                        Build from Claude Code, Claude.ai or ChatGPT
                        conversation exports
  --calibration-queries <file>
                        Real questions (JSON-lines) that validate the
                        calibration and are embedded as golden queries
  --calibration <file>  Use a prebuilt calibration instead of fitting one
  --skip-calibration    Ship the pack unscored
  --sentence-windows <n>
                        Also index every run of n consecutive sentences of
                        each chunk (2 recommended), so a one-sentence answer
                        is not averaged away by the rest of its chunk. Results
                        are still whole chunks. Costs roughly 8x the index
                        rows and build time; the pack cannot be extended
                        with append.

Custom encoder (advanced):
  --encoder-model <id>  HF model id to record in the artifact's
                        encoder declaration (default: sentence-transformers/
                        all-MiniLM-L6-v2). Requires --encoder-weights; the
                        model itself must match the kernel's compiled-in
                        shape (see export_encoder_blob.py's --model check).
  --encoder-weights <f> path to a weight blob exported by
                        export_encoder_blob.py for --encoder-model. Required
                        with --encoder-model.
  --encoder-vocab <f>   path to a WordPiece vocab.txt, if
                        different from the packaged bert-base-uncased table
                        (default: the packaged vocab.txt)
  --encoder-pooling mean|cls
                        mean (default) averages every token's
                        final hidden state; cls uses the [CLS] token (e.g.
                        Snowflake arctic-embed models). Must match how
                        --encoder-weights was exported. cls only pools the
                        first 512-token window of a longer document.
  --encoder-query-prefix <s>
                        text prepended to every query before
                        embedding (default: none). Some fine-tunes need one
                        — e.g. Snowflake arctic-embed models require
                        "Represent this sentence for searching relevant
                        passages: " on queries and nothing on passages, or
                        they silently lose the retrieval quality they were
                        trained for. Check --encoder-model's model card.
  --encoder-passage-prefix <s>
                        text prepended to every indexed
                        passage before embedding (default: none)`,
  mcp: `Usage:
  pikelet mcp --pack <file-or-url> [--pack ... | --shelf <file-or-url>] [--journal <name|dir>] [--no-cache]
  pikelet mcp install --pack <file-or-url> [--journal <name|dir>] [--client claude-code|claude-desktop]

mcp serves knowledge packs over the Model Context Protocol on stdio, so an
MCP client (Claude Code, Claude Desktop, an agent framework) can attach
them as a retrieval tool: search (per-pack results with provenance and
calibrated abstention), list_packs (names + immutable identities for
citation pinning), get_record, verify_pack (runs the golden queries and
abstention probes stored inside the pack). --pack takes a local file or an
HTTP(S) URL — URL packs are range-read, never downloaded whole — and
either form takes '#<sha256>' to pin the manifest identity (mismatches
refuse to serve). --shelf mounts every pack on a static packs.json listing
(see packs/README.md). mcp install writes the MCP client config instead of
running the server (--server-name names the entry, --force replaces it).

A URL pack's two largest mount reads, the resident index prefix and the
query encoder, are cached on disk under their sha256 in
~/.cache/pikelet/blobs ($PIKELET_CACHE_DIR or $XDG_CACHE_HOME moves it; 2 GiB,
least recently used evicted first), so a later process mounting the same
pack reads them locally. The verified pieces queries read (rows, records,
digest pages) are kept too, in logs capped at 64 MiB. Every hit is
hash-checked before use. --no-cache turns it off.

--journal mounts a private journal the agent may write to (docs/agent-journal.md):
write_note appends a note (handoff, decision, dead-end, measurement, summary,
todo, preference) as an immutable layer, and latest lists notes newest first.
Every other pack stays read-only. A bare name lives in ~/.pikelet/journals/;
journal init creates one (unscored, one seed record), journal list prints its
notes, journal compact folds its layers into one base (writes do this on
their own every 8 notes).`,
  append: `Usage:
  pikelet append --parent <file|url> [--parent ...] [--source <path|url> ...]
                 [--remove <id> ...] [--supersede <id>=<path> ...] --out layer.pikelet

${LAYER_TEXT}`,
  compact: `Usage:
  pikelet compact --head <base> [--head <layer> ...] --out compacted.pikelet [--no-refit]

${LAYER_TEXT}`,
  rebase: `Usage:
  pikelet rebase --layer <file> --onto <base> [--onto <layer> ...] --out rebased.pikelet
                 [--old-parent <file> ...] [--on-conflict refuse|skip|keep-both]
                 [--on-foreign refuse|drop]

${LAYER_TEXT}`,
  journal: `Usage:
  pikelet journal init <name|dir> [--force]
  pikelet journal list <name|dir> [--kind <kind>] [--tag <tag>] [--limit <n>]
  pikelet journal compact <name|dir>

--journal mounts a private journal the agent may write to (docs/agent-journal.md):
write_note appends a note (handoff, decision, dead-end, measurement, summary,
todo, preference) as an immutable layer, and latest lists notes newest first.
Every other pack stays read-only. A bare name lives in ~/.pikelet/journals/;
journal init creates one (unscored, one seed record), journal list prints its
notes, journal compact folds its layers into one base (writes do this on
their own every 8 notes).`,
  doctor: `Usage:
  pikelet doctor <url>

Probes how a web host serves a pack: Range requests and 206 responses,
cache-key ranges, HTTP/2, ETag and round-trip time. Exits 1 when a check
fails.`,
  create: `Usage:
  pikelet create --name my-docs-search --source ./docs --no-deploy --yes

Scaffolds a Cloudflare Worker search project from a folder or website, the
workflow from before single-file packs. For a .pikelet file, use
'pikelet compile'. With no flags, create asks for each answer.

Options:
  --name <dir>          Generated project directory
  --source <path|url>   Local folder or website URL
  --mode <mode>         Query embedding mode: workers-ai (default) or student.
                        student distills a corpus-specific encoder at build
                        time (needs Python with torch/transformers) and embeds
                        queries inside the Worker with no AI binding.
  --student-model <f>   Pre-trained PSTU model; skips training (--mode student)
  --student-vectors <f> Teacher document vectors (.f32) matching --student-model
  --student-abstention <f>  Calibrated abstention scorer for --student-model
  --epochs <n>          Student training epochs (--mode student)
  --skip-abstention     Train without the abstention scorer (--mode student)
  --model <id>          bge-small-en-v1.5
  --max-pages <n>       URL crawl cap
  --include <glob>      Folder include glob, repeatable
  --exclude <glob>      Folder exclude glob, repeatable
  --include-url <pat>   URL-path include pattern ('*' wildcard), repeatable
  --exclude-url <pat>   URL-path exclude pattern ('*' wildcard), repeatable;
                        aggregate pages like mdBook's print.html are always
                        excluded
  --runtime snapshot|artifact
                        artifact is deprecated: it serves the retired
                        .pikelet-range profile; prefer the default snapshot
                        runtime or 'compile' for a complete .pikelet
  --artifact <file>     Deprecated with --runtime artifact: prebuilt
                        .pikelet-range artifact
  --deploy / --no-deploy
  --yes
  --force               Replace an existing target directory`,
  rebuild: `Usage:
  pikelet rebuild --yes

Rebuilds the search assets of a project made with 'pikelet create', from
its pikelet.config.json.`,
};

function printHelp(command) {
  console.log(HELP[command] ?? HELP_OVERVIEW);
}

const BOOLEAN_FLAGS = new Set([
  'deploy', 'yes', 'force', 'verbose', 'skip-abstention', 'skip-calibration',
  'help', 'h',
  // append (LAYERED_PROFILE.md 6.1)
  'allow-drift',
  // compact (6.3): ship unscored instead of refitting. Boolean, so it does not
  // consume the next argument -- an unregistered flag here set itself to
  // '--out' and left --out unset.
  'no-refit',
  // mcp: skip the on-disk cache of URL packs' committed regions.
  'no-cache',
]);

// Every flag that takes a value. An unregistered flag is refused rather than
// guessed at: guessing that it takes a value swallowed the next argument
// (`mcp --no-resolve-parents pack.pikelet` ate the pack path and mounted
// nothing), and guessing boolean would silently ignore a typo.
const VALUE_FLAGS = new Set([
  'name', 'source', 'runtime', 'mode', 'artifact', 'out', 'license', 'model', 'epochs',
  'include', 'exclude', 'include-url', 'exclude-url', 'max-pages',
  'calibration', 'calibration-queries', 'git', 'git-rev', 'conversations', 'sentence-windows',
  'encoder-model', 'encoder-vocab', 'encoder-weights', 'encoder-pooling',
  'encoder-query-prefix', 'encoder-passage-prefix',
  'student-model', 'student-vectors', 'student-abstention',
  // mcp
  'pack', 'shelf', 'client', 'server-name', 'journal',
  // append / compact / rebase (LAYERED_PROFILE.md 6)
  'parent', 'parent-locator', 'layer', 'remove', 'supersede', 'on-conflict', 'on-foreign',
  'assert-ingest', 'reader-drift-limit', 'max-depth', 'head', 'onto', 'old-parent',
  // journal
  'kind', 'tag', 'limit',
]);

export function parseArgs(args) {
  const flags = {};
  // Flags that may appear more than once collect into an array. A flag that is
  // repeatable in spirit but missing here silently keeps only the LAST value,
  // which for append's --remove would drop deletions without a word.
  //
  // `source` collects like these, then collapses back to a string below for
  // every command but append: compile and create read it as a string
  // (`new URL(flags.source)`, `path.resolve(..., flags.source)`), while append
  // takes several. Leaving it out of this set kept only the last --source.
  const repeated = new Set(['include', 'exclude', 'include-url', 'exclude-url', 'pack', 'source',
    'remove', 'supersede',
    // compact takes a chain, listed base-first, so --head repeats.
    'head', 'parent',
    // rebase (6.2) names two chains: the new head and the layer's ORIGINAL
    // parent, each base-first.
    'onto', 'old-parent']);
  const positionals = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg.startsWith('--')) {
      positionals.push(arg);
      continue;
    }
    const name = arg.slice(2);
    // Boolean flags take no value. A flag missing here silently swallows the
    // NEXT argument as its value: `--allow-drift --out x` set
    // allow-drift="--out" and left --out unset.
    if (BOOLEAN_FLAGS.has(name)) {
      flags[name] = true;
      continue;
    }
    if (name === 'no-deploy') {
      flags.deploy = false;
      continue;
    }
    if (!VALUE_FLAGS.has(name)) throw new CliError(`Unknown flag --${name}\nNext: run pikelet --help for the flags each command takes`);
    if (i + 1 >= args.length) throw new CliError(`Missing value for --${name}`);
    const value = args[++i];
    if (repeated.has(name)) {
      if (!flags[name]) flags[name] = [];
      flags[name].push(value);
    } else {
      flags[name] = value;
    }
  }
  if (flags.source && positionals[0] !== 'append') {
    if (flags.source.length > 1) {
      throw new CliError(`--source may be given once${positionals[0] ? ` for ${positionals[0]}` : ''}; `
        + `got ${flags.source.length}`);
    }
    flags.source = flags.source[0];
  }
  return { flags, positionals };
}

async function createProject(flags) {
  const answers = await resolveCreateOptions(flags);
  const targetDir = path.resolve(process.cwd(), answers.name);
  const replacing = fssync.existsSync(targetDir);
  if (replacing) {
    if (!flags.force) throw new CliError(`Target directory already exists: ${targetDir}\nNext: rerun with --force or choose --name`);
    // --force replaces the project only after the new one has built, and
    // never a directory the run itself depends on.
    const within = (child, parent) => {
      const rel = path.relative(parent, child);
      return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
    };
    if (within(process.cwd(), targetDir)) {
      throw new CliError(`Refusing --force: ${targetDir} is the current directory or contains it\nNext: choose a --name outside it`);
    }
    if (!/^https?:\/\//i.test(answers.source)) {
      const sourceDir = path.resolve(process.cwd(), answers.source);
      if (within(sourceDir, targetDir)) {
        throw new CliError(`Refusing --force: the source ${sourceDir} is inside ${targetDir}, which --force would replace\nNext: move the source out or choose another --name`);
      }
      if (within(targetDir, sourceDir)) {
        throw new CliError(`Refusing --force: ${targetDir} is inside the source ${sourceDir}, so the project being replaced would be ingested\nNext: remove ${targetDir} first or choose a --name outside the source`);
      }
    }
  }

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-'));
  try {
    const config = makeConfig(answers);
    await writeProject(tmpDir, config);
    await stageStudentInputs(tmpDir, answers);
    await buildAssets(tmpDir, config, { sourceBaseDir: targetDir });
    await installProject(tmpDir, targetDir, replacing);
    console.log(`Scaffolded ${targetDir}`);
    console.log(`Rebuild: cd ${answers.name} && npm run reindex`);
    if (answers.deploy) await deployProject(targetDir);
  } catch (error) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

// Move the built project into place. The old project (with --force) is set
// aside first and restored if the move fails, so a failure at any point
// leaves it intact. The build dir lives in os.tmpdir(), often another
// filesystem than the target (WSL /tmp vs /mnt/c), where rename is EXDEV.
async function installProject(builtDir, targetDir, replacing) {
  const backup = replacing ? `${targetDir}.pikelet-old-${process.pid}` : null;
  if (backup) await fs.rename(targetDir, backup);
  try {
    try {
      await fs.rename(builtDir, targetDir);
    } catch (err) {
      if (err.code !== 'EXDEV') throw err;
      await fs.cp(builtDir, targetDir, { recursive: true });
      await fs.rm(builtDir, { recursive: true, force: true }).catch(() => {});
    }
  } catch (err) {
    await fs.rm(targetDir, { recursive: true, force: true }).catch(() => {});
    if (backup) await fs.rename(backup, targetDir);
    throw err;
  }
  if (backup) await fs.rm(backup, { recursive: true, force: true });
}

async function resolveCreateOptions(flags) {
  if (flags.yes && (!flags.name || !flags.source)) {
    throw new CliError('--yes requires --name and --source');
  }
  const rl = flags.yes ? null : readline.createInterface({ input, output });
  try {
    const name = flags.name || await rl.question('Project name: ');
    const source = flags.source || await rl.question('Path / URL: ');
    const deploy = flags.deploy !== undefined
      ? !!flags.deploy
      : flags.yes ? false : /^y/i.test(await rl.question('Deploy to Cloudflare when done? (y/N) '));
    if (!name || !source) throw new CliError('Project name and source are required');
    assertSourceFilterFlags(flags, /^https?:\/\//i.test(source));
    const mode = flags.mode || 'workers-ai';
    if (!['workers-ai', 'student'].includes(mode)) {
      throw new CliError(`--mode must be workers-ai or student, got ${mode}`);
    }
    if (mode !== 'student' && (flags['student-model'] || flags['student-vectors'] || flags['student-abstention'] || flags.epochs || flags['skip-abstention'])) {
      throw new CliError('--student-model, --student-vectors, --student-abstention, --epochs and --skip-abstention require --mode student');
    }
    if (flags['student-model'] && !flags['student-vectors']) {
      throw new CliError('--student-model requires --student-vectors so passage indexing uses matching teacher document vectors');
    }
    if (flags['student-vectors'] && !flags['student-model']) {
      throw new CliError('--student-vectors requires --student-model');
    }
    return {
      name,
      source,
      deploy,
      mode,
      studentModel: flags['student-model'] || null,
      studentVectors: flags['student-vectors'] || null,
      studentAbstention: flags['student-abstention'] || null,
      epochs: flags.epochs ? parsePositiveInt(flags.epochs, '--epochs') : null,
      skipAbstention: !!flags['skip-abstention'],
      model: flags.model || 'bge-small-en-v1.5',
      runtime: flags.runtime || (flags.artifact ? 'artifact' : 'snapshot'),
      artifact: flags.artifact || null,
      maxPages: parsePositiveInt(flags['max-pages'] || '500', '--max-pages'),
      include: flags.include || ['**/*.{md,mdx,html,txt}'],
      exclude: flags.exclude || ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
      includeUrl: flags['include-url'] || null,
      excludeUrl: flags['exclude-url'] || null,
    };
  } finally {
    rl?.close();
  }
}

function makeConfig(options) {
  const student = options.mode === 'student';
  const model = student ? null : MODEL_MAP[options.model];
  if (!student && !model) throw new CliError(`Unsupported --model ${options.model}. Supported: ${Object.keys(MODEL_MAP).join(', ')}`);
  if (!['snapshot', 'artifact'].includes(options.runtime)) {
    // 'complete' used to be accepted here and silently built a snapshot
    // project: the generated Worker templates only serve the snapshot and
    // artifact runtimes.
    throw new CliError(options.runtime === 'complete'
      ? 'The scaffold serves snapshot and artifact runtimes. To build a complete .pikelet artifact, run: pikelet compile --source <path|url> --out search.pikelet'
      : `--runtime must be snapshot or artifact, got ${options.runtime}`);
  }
  const isUrl = /^https?:\/\//i.test(options.source);
  const targetDir = path.resolve(process.cwd(), options.name);
  const artifactPath = options.artifact
    ? path.relative(targetDir, path.resolve(process.cwd(), options.artifact)) || '.'
    : null;
  return {
    $schema: CONFIG_SCHEMA_URL,
    version: 1,
    name: path.basename(options.name),
    source: isUrl
      ? {
          type: 'url',
          url: options.source,
          maxPages: options.maxPages,
          ...(options.includeUrl ? { includeUrl: options.includeUrl } : {}),
          ...(options.excludeUrl ? { excludeUrl: options.excludeUrl } : {}),
        }
      : { type: 'folder', path: path.relative(path.resolve(process.cwd(), options.name), path.resolve(process.cwd(), options.source)) || '.', include: options.include, exclude: options.exclude },
    chunking: { ...DEFAULT_CONFIG.chunking },
    embedding: student
      ? {
          mode: 'student',
          studentModelPath: options.studentModel ? 'student-model.bin' : 'student/student-model.bin',
          ...(options.studentVectors ? { teacherVectorsPath: 'docs-vectors.f32' } : {}),
          ...(options.studentModel
            ? {}
            : {
                trainStudent: {
                  enabled: true,
                  outDir: 'student',
                  ...(options.epochs ? { epochs: options.epochs } : {}),
                  ...(options.skipAbstention ? { skipAbstention: true } : {}),
                },
              }),
          dims: 384,
          prefixPolicy: { passage: '', query: '' },
          pooling: 'mean',
          normalize: true,
        }
      : { ...DEFAULT_CONFIG.embedding, buildModel: options.model, dims: model.dims },
    index: { ...DEFAULT_CONFIG.index },
    runtime: options.runtime === 'artifact'
      ? { mode: 'artifact', storage: 'bundled', ...(artifactPath ? { artifactPath } : {}) }
      : { ...DEFAULT_CONFIG.runtime },
    validation: { ...DEFAULT_CONFIG.validation },
  };
}

// The packaged inline MiniLM encoder inputs. vocab.txt ships in the tarball.
// The weight blob is npmignored: a repo checkout has it in this directory,
// and an installed package fetches it, digest-pinned, into the user model
// cache on first use (resolveInlineEncoderInputs in complete-build.mjs). The
// install directory may be read-only or purged (npx), so nothing is written
// there.
const INLINE_ENCODER_DIR = fileURLToPath(new URL('./inline-encoder/', import.meta.url));
function defaultInlineWeightsPath() {
  const packaged = path.join(INLINE_ENCODER_DIR, 'encoder-weights.bin');
  if (fssync.existsSync(packaged)) return packaged;
  return path.join(modelCacheDir(), 'all-MiniLM-L6-v2', 'encoder-weights.bin');
}

export async function compileArtifact(flags) {
  const sources = ['source', 'git', 'conversations'].filter((f) => flags[f] !== undefined);
  if (sources.length > 1) throw new CliError(`compile takes one of --source, --git or --conversations, got --${sources.join(' and --')}`);
  if (!sources.length) throw new CliError('compile requires --source <path|url>, --git <repo> or --conversations <file|folder>');
  if (flags['git-rev'] && !flags.git) throw new CliError('--git-rev requires --git');
  const sentenceWindowSize = flags['sentence-windows'] === undefined ? 0 : parsePositiveInt(flags['sentence-windows'], '--sentence-windows');
  if (sentenceWindowSize > 8) throw new CliError('--sentence-windows must be 8 or less (2 is the measured default)');
  const sourceFilters = ['include', 'exclude', 'include-url', 'exclude-url', 'max-pages'].filter((f) => flags[f] !== undefined);
  if (flags.git && sourceFilters.length) {
    throw new CliError('--include, --exclude, --include-url, --exclude-url and --max-pages filter --source; --git takes --git-rev');
  }
  if (flags.conversations && sourceFilters.length) {
    throw new CliError('--include, --exclude, --include-url, --exclude-url and --max-pages filter --source, not --conversations');
  }
  const unsupported = ['mode', 'runtime', 'artifact', 'deploy', 'student-model', 'student-vectors', 'student-abstention', 'epochs', 'skip-abstention', 'model']
    .filter((name) => flags[name] !== undefined);
  if (unsupported.length) {
    throw new CliError(`compile does not take --${unsupported.join(', --')}: it always builds a complete kind-3 .pikelet with the inline transformer encoder`);
  }
  if (flags.calibration && flags['skip-calibration']) {
    throw new CliError('--calibration and --skip-calibration are mutually exclusive');
  }
  // A non-default encoder needs its own weights: the packaged
  // encoder-weights.bin is MiniLM-L6's blob specifically, and the only
  // kind of mismatch that fails loudly is a wrong-sized blob (dim/vocab
  // mismatch) — same-shaped weights from a different model would load and
  // silently produce wrong embeddings. --encoder-pooling has no default
  // dependency on --encoder-model (a differently-pooled export of the
  // packaged model is valid), so it's not required here.
  if (flags['encoder-model'] && !flags['encoder-weights']) {
    throw new CliError('--encoder-model requires --encoder-weights (and --encoder-vocab if the '
      + 'vocabulary differs from the packaged bert-base-uncased WordPiece table): the packaged '
      + 'encoder-weights.bin is compiled from sentence-transformers/all-MiniLM-L6-v2 specifically.');
  }
  const encoderPooling = flags['encoder-pooling'] || 'mean';
  if (!['mean', 'cls'].includes(encoderPooling)) {
    throw new CliError(`--encoder-pooling must be mean or cls, got ${encoderPooling}`);
  }
  const outPath = path.resolve(process.cwd(), flags.out || 'search.pikelet');
  if (fssync.existsSync(outPath) && !flags.force) {
    throw new CliError(`Output file already exists: ${outPath}\nNext: rerun with --force or choose --out`);
  }
  const isGit = !!flags.git;
  const isConversations = !!flags.conversations;
  const isUrl = !isGit && !isConversations && /^https?:\/\//i.test(flags.source);
  if (!isGit && !isConversations) assertSourceFilterFlags(flags, isUrl);
  const name = flags.name
    || (isGit ? `${await gitRepoName(path.resolve(process.cwd(), flags.git))}-history`
      : isConversations ? 'conversations'
      : isUrl ? new URL(flags.source).hostname : path.basename(path.resolve(process.cwd(), flags.source)));
  const config = {
    $schema: CONFIG_SCHEMA_URL,
    version: 1,
    name,
    // Recorded in the artifact manifest (corpus.provenance.license) and
    // surfaced by mcp list_packs; packs meant for redistribution should
    // carry one.
    ...(flags.license ? { license: String(flags.license) } : {}),
    source: isGit
      ? { type: 'git', path: path.resolve(process.cwd(), flags.git), ...(flags['git-rev'] ? { rev: flags['git-rev'] } : {}) }
      : isConversations
      ? { type: 'conversations', path: path.resolve(process.cwd(), flags.conversations) }
      : isUrl
      ? {
          type: 'url',
          url: flags.source,
          maxPages: parsePositiveInt(flags['max-pages'] || '500', '--max-pages'),
          ...(flags['include-url'] ? { includeUrl: flags['include-url'] } : {}),
          ...(flags['exclude-url'] ? { excludeUrl: flags['exclude-url'] } : {}),
        }
      : {
          type: 'folder',
          path: path.resolve(process.cwd(), flags.source),
          include: flags.include || ['**/*.{md,mdx,html,txt}'],
          exclude: flags.exclude || ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
        },
    chunking: { ...DEFAULT_CONFIG.chunking },
    embedding: {
      mode: 'inline-transformer',
      dims: 384,
      // MiniLM-L6 needs no prefix. Some fine-tunes do — e.g. Snowflake's
      // arctic-embed family requires an asymmetric query prefix
      // ("Represent this sentence for searching relevant passages: ")
      // with no passage prefix; skipping it doesn't error, it just quietly
      // forfeits the retrieval quality the fine-tune was trained for; the
      // query() reader and calibration's embedQuery both read this from
      // the declaration automatically, so setting it here is sufficient.
      prefixPolicy: {
        passage: flags['encoder-passage-prefix'] || '',
        query: flags['encoder-query-prefix'] || '',
      },
      pooling: encoderPooling,
      normalize: true,
    },
    index: { ...DEFAULT_CONFIG.index, ...(sentenceWindowSize ? { sentenceWindows: sentenceWindowSize } : {}) },
    runtime: {
      mode: 'complete',
      profile: 'kind3',
      storage: 'bundled',
      fileName: path.basename(outPath),
      // Self-calibrated abstention by default; --calibration supplies a
      // prebuilt retrieval-signals asset instead, --skip-calibration ships
      // the unscored placeholder.
      ...(flags['skip-calibration'] || flags.calibration ? {} : { calibration: 'auto' }),
      // --calibration-queries: real, corpus-author-written questions used
      // ONLY as held-out validation and as the artifact's embedded golden
      // queries (see calibrate.mjs's loadCalibrationQueries and the v5
      // design note at the top of that file) — never as fit positives.
      // Resolved to an absolute path here, not left relative: compile's
      // projectDir is a temp directory deleted at the end of this
      // function, so a path relative to it would not survive past the
      // build even though loadCalibrationQueries always resolves relative
      // to projectDir for the scaffolded (create/rebuild) flow.
      ...(flags['calibration-queries']
        ? { calibrationQueries: path.resolve(process.cwd(), flags['calibration-queries']) }
        : {}),
      inlineEncoder: {
        vocabPath: flags['encoder-vocab']
          ? path.resolve(process.cwd(), flags['encoder-vocab'])
          : path.join(INLINE_ENCODER_DIR, 'vocab.txt'),
        weightsPath: flags['encoder-weights']
          ? path.resolve(process.cwd(), flags['encoder-weights'])
          : defaultInlineWeightsPath(),
        model: flags['encoder-model'] || 'sentence-transformers/all-MiniLM-L6-v2',
        maxTokens: 512,
        ...(flags.calibration ? { calibrationPath: path.resolve(process.cwd(), flags.calibration) } : {}),
      },
    },
    validation: { ...DEFAULT_CONFIG.validation },
  };
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-compile-'));
  try {
    await buildAssets(tmpDir, config, { skipBundleSizeCheck: true });
    const manifest = JSON.parse(await fs.readFile(path.join(tmpDir, 'assets', 'manifest.json'), 'utf8'));
    await fs.mkdir(path.dirname(outPath), { recursive: true });
    // copyFile, not rename: os.tmpdir() and the output may be different mounts.
    await fs.copyFile(path.join(tmpDir, 'assets', config.runtime.fileName), outPath);
    console.log(`Compiled ${outPath}`);
    console.log(`  ${(manifest.artifact.bytes / 1024 / 1024).toFixed(2)} MB, ${manifest.chunkCount} records, identity ${manifest.artifact.identity}`);
    console.log("Query it from any runtime: openPikeletFile from 'pikelet-wasm/complete'");
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

async function rebuildProject(projectDir, flags = {}) {
  const configPath = path.join(projectDir, 'pikelet.config.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  applyRuntimeOverrides(config, projectDir, flags);
  validateConfig(config);
  await writeRuntimeModules(projectDir, config);
  await fs.writeFile(path.join(projectDir, 'wrangler.toml'), wranglerToml(config));
  await buildAssets(projectDir, config, { verbose: !!flags.verbose });
  await fs.writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`);
  console.log('Rebuilt Pikelet search assets.');
}
function applyRuntimeOverrides(config, projectDir, flags) {
  if (!flags.runtime && !flags.artifact) return;
  const runtime = flags.runtime || (flags.artifact ? 'artifact' : config.runtime?.mode || 'snapshot');
  if (!['snapshot', 'artifact'].includes(runtime)) {
    throw new CliError(runtime === 'complete'
      ? 'The scaffold serves snapshot and artifact runtimes. To build a complete .pikelet artifact, run: pikelet compile --source <path|url> --out search.pikelet'
      : `--runtime must be snapshot or artifact, got ${runtime}`);
  }
  if (runtime === 'artifact') {
    const artifactPath = flags.artifact
      ? path.relative(projectDir, path.resolve(process.cwd(), flags.artifact)) || '.'
      : config.runtime?.artifactPath;
    config.runtime = { mode: 'artifact', storage: 'bundled', ...(artifactPath ? { artifactPath } : {}) };
  } else {
    config.runtime = { ...DEFAULT_CONFIG.runtime };
  }
}
// --include/--exclude are filesystem globs over a folder source;
// --include-url/--exclude-url are '*'-wildcard URL-path patterns over a
// crawl. Passing one family against the other source type used to be
// silently ignored; erroring guarantees no filter ever no-ops.
function assertSourceFilterFlags(flags, isUrl) {
  if (isUrl && (flags.include || flags.exclude)) {
    throw new CliError("--include/--exclude are folder globs and do not apply to a URL source; use --include-url/--exclude-url ('*'-wildcard URL path patterns, e.g. --exclude-url '*/print.html')");
  }
  if (!isUrl && (flags['include-url'] || flags['exclude-url'])) {
    throw new CliError('--include-url/--exclude-url apply only to a URL source; use --include/--exclude folder globs');
  }
}
function parsePositiveInt(value, name) {
  const parsed = parseInt(String(value), 10);
  if (!Number.isInteger(parsed) || parsed < 1) throw new CliError(`${name} must be a positive integer`);
  return parsed;
}

// Re-exported for consumers that import the package internals (the Docusaurus
// plugin imports these from ../src/cli.mjs).
export { buildSearchAssets } from './build.mjs';
export { fetchInlineEncoderWeights } from './complete-build.mjs';
export { CliError } from './common.mjs';
