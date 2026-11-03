import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');

export class PikeletMcpClient {
  constructor({ packs, extraArgs = [], timeoutMs = 60_000 }) {
    const bin = path.join(REPO_ROOT, 'packages', 'pikelet', 'bin', 'pikelet.mjs');
    const args = ['mcp', ...packs.flatMap((p) => ['--pack', p]), ...extraArgs];
    this.timeoutMs = timeoutMs;
    this.nextId = 1;
    this.messages = [];
    this.buffer = '';
    this.child = spawn(process.execPath, [bin, ...args], { stdio: ['pipe', 'pipe', 'inherit'] });
    this.child.stdout.on('data', (d) => this.#onData(d));
  }

  #onData(d) {
    this.buffer += d;
    let i;
    while ((i = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, i);
      this.buffer = this.buffer.slice(i + 1);
      if (!line.trim()) continue;
      try { this.messages.push(JSON.parse(line)); }
      catch { console.error('Bad MCP JSON line:', line); }
    }
  }

  send(msg) { this.child.stdin.write(JSON.stringify(msg) + '\n'); }

  wait(id) {
    return new Promise((resolve, reject) => {
      const deadline = Date.now() + this.timeoutMs;
      const timer = setInterval(() => {
        const idx = this.messages.findIndex((m) => m.id === id);
        if (idx >= 0) {
          const [msg] = this.messages.splice(idx, 1);
          clearInterval(timer);
          resolve(msg);
        } else if (Date.now() > deadline) {
          clearInterval(timer);
          reject(new Error(`MCP timeout waiting for id ${id}`));
        }
      }, 10);
    });
  }

  async initialize(name = 'pikelet-example-client') {
    const id = this.nextId++;
    this.send({
      jsonrpc: '2.0', id, method: 'initialize',
      params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name, version: '0' } },
    });
    const msg = await this.wait(id);
    if (msg.error) throw new Error(JSON.stringify(msg.error));
    this.send({ jsonrpc: '2.0', method: 'notifications/initialized' });
    return msg.result;
  }

  async call(name, args = {}) {
    const id = this.nextId++;
    this.send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
    const msg = await this.wait(id);
    if (msg.error) throw new Error(JSON.stringify(msg.error));
    const text = msg.result?.content?.[0]?.text;
    if (typeof text !== 'string') return msg.result;
    try { return JSON.parse(text); } catch { return text; }
  }

  close() {
    try { this.child.stdin.end(); } catch { /* already closed */ }
    setTimeout(() => this.child.kill(), 300);
  }
}
