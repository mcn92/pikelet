// Generated-project writing: runtime modules, templates, wrangler.toml and
// package.json generation, student input staging, and deploy.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { PACKAGE_ROOT, CLI_VERSION, PIKELET_WASM_RANGE, CliError } from './common.mjs';

async function writeProject(projectDir, config) {
  await fs.mkdir(path.join(projectDir, 'assets'), { recursive: true });
  await fs.mkdir(path.join(projectDir, '.github', 'workflows'), { recursive: true });
  await fs.writeFile(path.join(projectDir, 'pikelet.config.json'), `${JSON.stringify(config, null, 2)}\n`);
  await writeRuntimeModules(projectDir, config);
  await copyTemplate('ui.html', path.join(projectDir, 'ui.html'));
  await copyTemplate('README.generated.md', path.join(projectDir, 'README.md'));
  await copyTemplate('reindex.yml', path.join(projectDir, '.github', 'workflows', 'reindex.yml'));
  await fs.writeFile(path.join(projectDir, 'wrangler.toml'), wranglerToml(config));
  await fs.writeFile(path.join(projectDir, 'wrangler.local.toml'), wranglerToml(config, { localStubAi: true }));
  await fs.writeFile(path.join(projectDir, 'package.json'), generatedPackageJson(config));
  await fs.writeFile(path.join(projectDir, '.gitignore'), 'node_modules\n.wrangler\n.pikelet/last-build.log\n');
}

async function copyTemplate(name, dest) {
  await fs.copyFile(path.join(PACKAGE_ROOT, 'templates', name), dest);
}

async function writeRuntimeModules(projectDir, config) {
  const student = config.embedding?.mode === 'student';
  await copyTemplate(config.runtime?.mode === 'artifact' ? 'worker.artifact.js' : 'worker.js', path.join(projectDir, 'worker.js'));
  await copyTemplate(student ? 'encoder.student.js' : 'encoder.workers-ai.js', path.join(projectDir, 'encoder.js'));
  if (student) {
    await fs.copyFile(path.join(PACKAGE_ROOT, 'src', 'student-embedder.mjs'), path.join(projectDir, 'student-embedder.mjs'));
  }
}

async function stageStudentInputs(projectDir, options) {
  if (options.mode !== 'student' || !options.studentModel) return;
  const modelPath = path.resolve(process.cwd(), options.studentModel);
  if (!fssync.existsSync(modelPath)) throw new CliError(`Student model not found: ${modelPath}`);
  await fs.copyFile(modelPath, path.join(projectDir, 'student-model.bin'));
  const vectorsPath = path.resolve(process.cwd(), options.studentVectors);
  if (!fssync.existsSync(vectorsPath)) throw new CliError(`Teacher vectors not found: ${vectorsPath}`);
  await fs.copyFile(vectorsPath, path.join(projectDir, 'docs-vectors.f32'));
  if (options.studentAbstention) {
    const abstentionPath = path.resolve(process.cwd(), options.studentAbstention);
    if (!fssync.existsSync(abstentionPath)) throw new CliError(`Abstention model not found: ${abstentionPath}`);
    await fs.copyFile(abstentionPath, path.join(projectDir, 'student-abstention.json'));
  }
}

function wranglerToml(config, options = {}) {
  const student = config.embedding?.mode === 'student';
  const localStubAi = options.localStubAi === true;
  return `name = "${tomlString(config.name)}"
main = "worker.js"
compatibility_date = "2025-04-09"
compatibility_flags = ["nodejs_compat"]
${student || localStubAi ? '' : `
[ai]
binding = "AI"
`}
[vars]
READ_ONLY = "1"
RATE_LIMIT_RPM = "120"
MAX_JSON_BYTES = "262144"
MAX_QUERY_CHARS = "4096"
${student ? '' : `LOCAL_STUB_AI = "${localStubAi ? '1' : '0'}"\n`}
[[rules]]
type = "Data"
globs = ["**/*.pnck", "**/*.pikelet-range", "**/*.bin"]
fallthrough = true

[[rules]]
type = "Text"
globs = ["ui.html"]
fallthrough = true
`;
}

function generatedPackageJson(config) {
  return `${JSON.stringify({
    name: config.name,
    version: '0.1.0',
    private: true,
    type: 'module',
    scripts: {
      dev: 'wrangler dev --config wrangler.local.toml',
      deploy: 'wrangler deploy',
      reindex: 'pikelet rebuild --yes',
    },
    dependencies: {
      'pikelet-wasm': PIKELET_WASM_RANGE,
    },
    devDependencies: {
      'pikelet': CLI_VERSION,
      wrangler: '^4.81.1',
    },
  }, null, 2)}\n`;
}

function tomlString(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]/g, '-').replace(/^-+|-+$/g, '') || 'pikelet-search';
}
async function deployProject(projectDir) {
  const who = runNpmSync(['exec', '--', 'wrangler', 'whoami'], { cwd: projectDir, encoding: 'utf8' });
  if (who.status !== 0) {
    throw new CliError(`Cloudflare auth required. Next: cd ${projectDir} && npx wrangler login && npm run deploy`, 3);
  }
  const deploy = runNpmSync(['exec', '--', 'wrangler', 'deploy'], { cwd: projectDir, encoding: 'utf8' });
  if (deploy.status !== 0) {
    throw new CliError(`Deploy failed.\n${deploy.stderr || deploy.stdout}\nNext: cd ${projectDir} && npm run deploy`, 3);
  }
  process.stdout.write(deploy.stdout);
}

function npmCliPath() {
  if (process.env.npm_execpath && process.env.npm_execpath.endsWith('.js')) {
    return { command: process.execPath, args: [process.env.npm_execpath], shell: false };
  }
  if (process.platform === 'win32') return { command: 'cmd.exe', args: ['/d', '/s', '/c', 'npm'], shell: false };
  return { command: 'npm', args: [], shell: false };
}

function runNpmSync(args, options) {
  const npm = npmCliPath();
  return spawnSync(npm.command, [...npm.args, ...args], { ...options, shell: npm.shell });
}

export {
  writeProject,
  copyTemplate,
  writeRuntimeModules,
  stageStudentInputs,
  wranglerToml,
  generatedPackageJson,
  tomlString,
  deployProject,
  npmCliPath,
  runNpmSync,
};
