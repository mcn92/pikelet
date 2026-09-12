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
import { CONFIG_SCHEMA_URL, DEFAULT_CONFIG, MODEL_MAP, CliError } from './common.mjs';
import { writeProject, writeRuntimeModules, stageStudentInputs, wranglerToml, deployProject } from './scaffold.mjs';
import { buildAssets, validateConfig } from './build.mjs';

export async function main(argv) {
  const parsed = parseArgs(argv.slice(2));
  if (parsed.positionals[0] === 'doctor') {
    const { runDoctor } = await import('./doctor.mjs');
    const report = await runDoctor(parsed.positionals[1]);
    if (report.failures) {
      throw Object.assign(new Error('doctor found failing hosting checks'), { exitCode: 1 });
    }
    return;
  }
  if (parsed.positionals[0] === 'mcp') {
    const { runMcpServer, loadShelf, installMcpConfig } = await import('./mcp.mjs');
    const { loadCompleteModules, CLI_VERSION } = await import('./common.mjs');
    // Positional pack paths (and URLs, optionally `#sha256`-pinned)
    // compose with repeated --pack; --shelf mounts a static pack listing.
    const positionalPacks = parsed.positionals.slice(1).filter((p) => p !== 'install');
    const packPaths = [...(parsed.flags.pack || []), ...positionalPacks];
    if (parsed.positionals[1] === 'install') {
      const written = await installMcpConfig({
        packPaths,
        shelf: parsed.flags.shelf,
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
      packPaths.push(...await loadShelf(parsed.flags.shelf));
    }
    const { reader } = await loadCompleteModules();
    await runMcpServer({
      packPaths,
      openPikeletFile: reader.openPikeletFile,
      httpRangeSource: reader.httpRangeSource,
      serverVersion: CLI_VERSION,
    });
    return;
  }
  // `pikelet create` is the canonical form; bare flags still scaffold for
  // compatibility with the old `npm create` flow.
  const command = ['rebuild', 'compile', 'create'].includes(parsed.positionals[0]) ? parsed.positionals[0] : 'create';
  if (parsed.flags.help || parsed.flags.h) {
    printHelp();
    return;
  }
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

function printHelp() {
  console.log(`pikelet

Usage:
  npx pikelet create --name my-docs-search --source ./docs --no-deploy --yes
  pikelet compile --source <path|url> --out search.pikelet
  pikelet rebuild --yes
  pikelet doctor <url>   # probe artifact hosting: Range/206, cache-key ranges, h2, ETag, RTT
  pikelet mcp --pack <file-or-url> [--pack ... | --shelf <file-or-url>]
  pikelet mcp install --pack <file-or-url> [--client claude-code|claude-desktop]

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

compile builds a complete kind-3 .pikelet artifact from the source and stops:
no project, no Worker, no Cloudflare. The file carries the corpus, index,
inline MiniLM query encoder, calibrated abstention, and evaluation data; open
it with pikelet-wasm/complete on any runtime. Abstention is self-calibrated
from the corpus at build time (skipped with a logged reason when the corpus
cannot support a trustworthy fit); --calibration <file> supplies a prebuilt
retrieval-signals asset instead, --skip-calibration ships the artifact
unscored. compile also takes --source, --out, --name, --max-pages, --include,
--exclude, --force (overwrite the output file), and --encoder-model/
--encoder-weights/--encoder-vocab/--encoder-pooling/--encoder-query-prefix/
--encoder-passage-prefix to swap the packaged MiniLM-L6 encoder for another
BERT-shaped one built the same way (see
examples/05-one-file-search/encoder-spike/export_encoder_blob.py) — the
compiled kernel's dimensions are fixed at build time, so a swapped model
must match MiniLM-L6's shape (384 hidden / 6 layers / 12 heads / 1536 FFN /
30522 vocab) exactly.

Flags:
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
  --encoder-model <id>  compile only: HF model id to record in the artifact's
                        encoder declaration (default: sentence-transformers/
                        all-MiniLM-L6-v2). Requires --encoder-weights; the
                        model itself must match the kernel's compiled-in
                        shape (see export_encoder_blob.py's --model check).
  --encoder-weights <f> compile only: path to a weight blob exported by
                        export_encoder_blob.py for --encoder-model. Required
                        with --encoder-model.
  --encoder-vocab <f>   compile only: path to a WordPiece vocab.txt, if
                        different from the packaged bert-base-uncased table
                        (default: the packaged vocab.txt)
  --encoder-pooling mean|cls
                        compile only: mean (default) averages every token's
                        final hidden state; cls uses the [CLS] token (e.g.
                        Snowflake arctic-embed models). Must match how
                        --encoder-weights was exported. cls only pools the
                        first 512-token window of a longer document.
  --encoder-query-prefix <s>
                        compile only: text prepended to every query before
                        embedding (default: none). Some fine-tunes need one
                        — e.g. Snowflake arctic-embed models require
                        "Represent this sentence for searching relevant
                        passages: " on queries and nothing on passages, or
                        they silently lose the retrieval quality they were
                        trained for. Check --encoder-model's model card.
  --encoder-passage-prefix <s>
                        compile only: text prepended to every indexed
                        passage before embedding (default: none)
  --runtime snapshot|artifact
                        artifact is deprecated: it serves the retired
                        .pikelet-range profile; prefer the default snapshot
                        runtime or 'compile' for a complete .pikelet
  --artifact <file>     Deprecated with --runtime artifact: prebuilt
                        .pikelet-range artifact
  --out <file>          compile only: output path (default search.pikelet)
  --deploy / --no-deploy
  --yes
  --force               Replace an existing target directory (or output file for compile)
`);
}

function parseArgs(args) {
  const flags = {};
  const repeated = new Set(['include', 'exclude', 'include-url', 'exclude-url', 'pack']);
  const positionals = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg.startsWith('--')) {
      positionals.push(arg);
      continue;
    }
    const name = arg.slice(2);
    if (name === 'deploy' || name === 'yes' || name === 'force' || name === 'skip-abstention' || name === 'skip-calibration' || name === 'help' || name === 'h') {
      flags[name] = true;
      continue;
    }
    if (name === 'no-deploy') {
      flags.deploy = false;
      continue;
    }
    if (i + 1 >= args.length) throw new CliError(`Missing value for --${name}`);
    const value = args[++i];
    if (repeated.has(name)) {
      if (!flags[name]) flags[name] = [];
      flags[name].push(value);
    } else {
      flags[name] = value;
    }
  }
  return { flags, positionals };
}

async function createProject(flags) {
  const answers = await resolveCreateOptions(flags);
  const targetDir = path.resolve(process.cwd(), answers.name);
  if (fssync.existsSync(targetDir)) {
    if (!flags.force) throw new CliError(`Target directory already exists: ${targetDir}\nNext: rerun with --force or choose --name`);
    await fs.rm(targetDir, { recursive: true, force: true });
  }

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-'));
  try {
    const config = makeConfig(answers);
    await writeProject(tmpDir, config);
    await stageStudentInputs(tmpDir, answers);
    await buildAssets(tmpDir, config, { sourceBaseDir: targetDir });
    await fs.rename(tmpDir, targetDir);
    console.log(`Scaffolded ${targetDir}`);
    console.log(`Rebuild: cd ${answers.name} && npm run reindex`);
    if (answers.deploy) await deployProject(targetDir);
  } catch (error) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
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

// The packaged inline MiniLM encoder inputs. vocab.txt ships in the tarball;
// the weight blob is npmignored and auto-fetched into this directory on first
// use, digest-pinned (resolveInlineEncoderInputs in complete-build.mjs).
const INLINE_ENCODER_DIR = fileURLToPath(new URL('./inline-encoder/', import.meta.url));

async function compileArtifact(flags) {
  if (!flags.source) throw new CliError('compile requires --source <path|url>');
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
  const isUrl = /^https?:\/\//i.test(flags.source);
  assertSourceFilterFlags(flags, isUrl);
  const name = flags.name
    || (isUrl ? new URL(flags.source).hostname : path.basename(path.resolve(process.cwd(), flags.source)));
  const config = {
    $schema: CONFIG_SCHEMA_URL,
    version: 1,
    name,
    // Recorded in the artifact manifest (corpus.provenance.license) and
    // surfaced by mcp list_packs; packs meant for redistribution should
    // carry one.
    ...(flags.license ? { license: String(flags.license) } : {}),
    source: isUrl
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
    index: { ...DEFAULT_CONFIG.index },
    runtime: {
      mode: 'complete',
      profile: 'kind3',
      storage: 'bundled',
      fileName: path.basename(outPath),
      // Self-calibrated abstention by default; --calibration supplies a
      // prebuilt retrieval-signals asset instead, --skip-calibration ships
      // the unscored placeholder.
      ...(flags['skip-calibration'] || flags.calibration ? {} : { calibration: 'auto' }),
      inlineEncoder: {
        vocabPath: flags['encoder-vocab']
          ? path.resolve(process.cwd(), flags['encoder-vocab'])
          : path.join(INLINE_ENCODER_DIR, 'vocab.txt'),
        weightsPath: flags['encoder-weights']
          ? path.resolve(process.cwd(), flags['encoder-weights'])
          : path.join(INLINE_ENCODER_DIR, 'encoder-weights.bin'),
        model: flags['encoder-model'] || 'sentence-transformers/all-MiniLM-L6-v2',
        maxTokens: 512,
        ...(flags.calibration ? { calibrationPath: path.resolve(process.cwd(), flags.calibration) } : {}),
      },
    },
    validation: { ...DEFAULT_CONFIG.validation },
  };
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'pancake-compile-'));
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
