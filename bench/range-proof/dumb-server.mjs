// A deliberately dumb static file server: fs.createReadStream + Range
// header support, nothing else. No index, no query parsing, no knowledge
// of what a .pikelet is. Every request's Range header is logged so the
// proof can show exactly what Pikelet asked for.
//
// Usage: node dumb-server.mjs <dir> [port]
// Also importable as startDumbServer(dir, { port, logToStdout }) for the
// demo script to run in-process and capture the log.

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

export function startDumbServer(dir, { port = 0, logToStdout = true } = {}) {
  const log = [];
  const rootDir = path.resolve(dir);
  const server = http.createServer((req, res) => {
    const urlPath = new URL(req.url, 'http://localhost').pathname;
    const filePath = path.join(rootDir, decodeURIComponent(urlPath));
    if (!filePath.startsWith(rootDir)) {
      res.writeHead(403).end();
      return;
    }
    let stat;
    try {
      stat = fs.statSync(filePath);
    } catch {
      res.writeHead(404).end('not found');
      return;
    }

    const range = req.headers.range;
    const entry = { method: req.method, path: urlPath, range: range || null, time: Date.now() };
    log.push(entry);
    if (logToStdout) {
      console.log(`${req.method} ${urlPath}`);
      if (range) {
        console.log(`Range: ${range}`);
      } else if (req.method === 'HEAD') {
        console.log('(HEAD probe, no body sent)');
      } else {
        console.log('(no Range header — full file request)');
      }
    }

    if (req.method === 'HEAD') {
      res.writeHead(200, {
        'Content-Length': stat.size,
        'Accept-Ranges': 'bytes',
        'ETag': `"${stat.size}-${stat.mtimeMs}"`,
      });
      res.end();
      return;
    }

    if (!range) {
      res.writeHead(200, {
        'Content-Length': stat.size,
        'Accept-Ranges': 'bytes',
        'ETag': `"${stat.size}-${stat.mtimeMs}"`,
      });
      fs.createReadStream(filePath).pipe(res);
      return;
    }

    const m = /^bytes=(\d+)-(\d+)$/.exec(range);
    if (!m) {
      res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` }).end();
      return;
    }
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), stat.size - 1);
    res.writeHead(206, {
      'Content-Length': end - start + 1,
      'Content-Range': `bytes ${start}-${end}/${stat.size}`,
      'Accept-Ranges': 'bytes',
      'ETag': `"${stat.size}-${stat.mtimeMs}"`,
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
  });

  return new Promise((resolve) => {
    server.listen(port, '127.0.0.1', () => {
      const actualPort = server.address().port;
      resolve({
        url: `http://127.0.0.1:${actualPort}`,
        log,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}

// CLI mode: `node dumb-server.mjs <dir> [port]`
if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = process.argv[2] || '.';
  const port = Number(process.argv[3]) || 8000;
  const { url } = await startDumbServer(dir, { port });
  console.log(`serving ${dir} on ${url}`);
  console.log('(plain fs.createReadStream + Range support — no database, no index, no knowledge of .pikelet)');
}
