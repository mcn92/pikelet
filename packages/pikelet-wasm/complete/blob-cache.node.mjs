// A disk store for openPikeletFile's options.blobCache: bytes a pack commits
// to by sha256, filed under that digest. Two kinds of entry reach it:
//
//   - whole regions read at mount (the sketch's resident prefix, the
//     query-interp segment holding the encoder): most of what a URL mount
//     downloads, about 71 of 77 MiB on the 649 MiB Simple English Wikipedia
//     pack. One file each.
//   - pieces read per query (rerank vector rows, corpus records, and the
//     digest pages that verify them): a few hundred bytes to a few KiB, and
//     hundreds per query. One file each would be a poor fit, so entries under
//     SMALL_ENTRY_BYTES go into append-only log files (generations) with an
//     in-memory index, and whole generations rotate out under their own
//     budget.
//
// The store is untrusted by design: the reader hashes every hit against the
// digest it asked for and treats a mismatch as a miss (blobCacheGet in
// pikelet-artifact-common.js), so a torn write, a corrupted file, or two
// processes appending to one log at once costs a refetch, never a wrong byte.
// Entries are immutable and never go stale; the only policy is size.

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

// A full sha256, or a truncated one of at least 128 bits (sketch row digests
// may be 16-32 bytes).
const KEY = /^(?:[0-9a-f]{2}){16,32}$/;
export const DEFAULT_BLOB_CACHE_BYTES = 2 * 1024 * 1024 * 1024;
export const DEFAULT_SMALL_CACHE_BYTES = 64 * 1024 * 1024;
// Entries below this go to the log; at or above it, a file of their own.
export const SMALL_ENTRY_BYTES = 64 * 1024;
// The log keeps this many generations; the oldest is deleted whole when the
// newest fills (smallMaxBytes / GENERATIONS each).
const GENERATIONS = 4;
// Log record: magic 'PKB1' (u32 LE), key length (u8), key bytes, data length
// (u32 LE), data. A scan stops at the first record that does not parse.
const RECORD_MAGIC = 0x3142_4b50;
const GEN_FILE = /^gen-(\d{8})\.log$/;
// Entries are pack content, record text included, and a pack can be a
// private repository or a conversation history: the store is private to its
// owner whatever the umask.
const DIR_MODE = 0o700;
const FILE_MODE = 0o600;

// $PIKELET_CACHE_DIR, else $XDG_CACHE_HOME/pikelet/blobs, else
// ~/.cache/pikelet/blobs: the same root the CLI already caches encoder
// weights under.
export function defaultBlobCacheDir(env = process.env) {
    if (env.PIKELET_CACHE_DIR) return path.resolve(env.PIKELET_CACHE_DIR, 'blobs');
    const base = env.XDG_CACHE_HOME ? path.resolve(env.XDG_CACHE_HOME) : path.join(os.homedir(), '.cache');
    return path.join(base, 'pikelet', 'blobs');
}

export function diskBlobCache(dir = defaultBlobCacheDir(), { maxBytes = DEFAULT_BLOB_CACHE_BYTES, smallMaxBytes = DEFAULT_SMALL_CACHE_BYTES } = {}) {
    for (const [label, value] of [['maxBytes', maxBytes], ['smallMaxBytes', smallMaxBytes]]) {
        if (!Number.isSafeInteger(value) || value < 0) {
            throw new Error(`diskBlobCache ${label} must be a non-negative integer`);
        }
    }
    const root = path.resolve(dir);
    const checkKey = (key) => {
        if (typeof key !== 'string' || !KEY.test(key)) {
            throw new Error(`blob cache key must be a lowercase sha256 hex digest (32-64 hex digits), got ${JSON.stringify(key)}`);
        }
        return key;
    };
    const fileOf = (key) => path.join(root, key.slice(0, 2), checkKey(key));
    // Once per instance, before the first write: a root made by an older
    // store (or by mkdir -p under umask 022) is tightened, which also closes
    // off whatever it already holds. Best effort; on Windows modes are moot.
    let privateRoot = null;
    const makeRootPrivate = () => (privateRoot ??= (async () => {
        await fs.mkdir(root, { recursive: true, mode: DIR_MODE });
        await fs.chmod(root, DIR_MODE).catch(() => {});
    })());
    const pending = new Set();
    const track = (promise) => {
        pending.add(promise);
        promise.finally(() => pending.delete(promise)).catch(() => {});
        return promise;
    };

    const evict = async (keep) => {
        const entries = [];
        let total = 0;
        for (const shard of await fs.readdir(root).catch(() => [])) {
            if (shard === 'small') continue;
            const shardDir = path.join(root, shard);
            for (const name of await fs.readdir(shardDir).catch(() => [])) {
                if (!KEY.test(name)) continue;
                const file = path.join(shardDir, name);
                const st = await fs.stat(file).catch(() => null);
                if (!st) continue;
                total += st.size;
                entries.push({ file, size: st.size, mtime: st.mtimeMs });
            }
        }
        entries.sort((a, b) => a.mtime - b.mtime);
        for (const entry of entries) {
            if (total <= maxBytes) break;
            if (entry.file === keep) continue;
            await fs.rm(entry.file, { force: true });
            total -= entry.size;
        }
    };

    // --- the small-entry log -------------------------------------------------
    const smallDir = path.join(root, 'small');
    const genBytes = Math.max(1, Math.floor(smallMaxBytes / GENERATIONS));
    const genPath = (gen) => path.join(smallDir, `gen-${String(gen).padStart(8, '0')}.log`);
    const index = new Map(); // key -> { gen, offset, length }
    const handles = new Map(); // gen -> Promise<FileHandle> (read)
    let current = 0;
    // Set when the newest generation ends in bytes that do not parse (a torn
    // write): a scan stops there, so anything appended after it would never
    // be indexed again. Writing resumes in a fresh generation instead.
    let sealed = false;
    let queue = [];
    let flushing = null;

    const listGens = async () => (await fs.readdir(smallDir).catch(() => []))
        .map((name) => GEN_FILE.exec(name)).filter(Boolean).map((m) => Number(m[1])).sort((a, b) => a - b);

    const forgetGen = async (gen) => {
        for (const [key, at] of index) if (at.gen === gen) index.delete(key);
        const handle = handles.get(gen);
        handles.delete(gen);
        if (handle) await handle.then((h) => h.close(), () => {});
    };

    const scan = (gen, bytes) => {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        let at = 0;
        while (at + 9 <= bytes.length && view.getUint32(at, true) === RECORD_MAGIC) {
            const keyLen = bytes[at + 4];
            if (keyLen < 16 || keyLen > 32 || at + 5 + keyLen + 4 > bytes.length) break;
            const key = Buffer.from(bytes.subarray(at + 5, at + 5 + keyLen)).toString('hex');
            const length = view.getUint32(at + 5 + keyLen, true);
            const offset = at + 9 + keyLen;
            if (offset + length > bytes.length) break;
            index.set(key, { gen, offset, length });
            at = offset + length;
        }
        return at === bytes.length;
    };

    // Built once, on first use: every generation still on disk is read and
    // indexed (at most smallMaxBytes), later generations overriding earlier.
    let ready = null;
    const smallReady = () => (ready ??= (async () => {
        if (smallMaxBytes === 0) return;
        const gens = await listGens();
        for (const gen of gens.slice(0, Math.max(0, gens.length - GENERATIONS))) {
            await fs.rm(genPath(gen), { force: true });
        }
        for (const gen of gens.slice(-GENERATIONS)) {
            const bytes = await fs.readFile(genPath(gen)).catch(() => null);
            const whole = bytes ? scan(gen, bytes) : true;
            if (gen === gens[gens.length - 1]) sealed = !whole;
        }
        current = gens.length ? gens[gens.length - 1] : 0;
    })());

    const readSmall = async (key) => {
        const at = index.get(key);
        if (!at) return null;
        let handle = handles.get(at.gen);
        if (!handle) {
            handle = fs.open(genPath(at.gen), 'r');
            handles.set(at.gen, handle);
            handle.catch(() => { if (handles.get(at.gen) === handle) handles.delete(at.gen); });
        }
        let fh;
        try { fh = await handle; } catch (err) {
            // Another process rotated this generation away.
            if (err?.code === 'ENOENT') { await forgetGen(at.gen); return null; }
            throw err;
        }
        const out = new Uint8Array(at.length);
        const { bytesRead } = await fh.read(out, 0, at.length, at.offset);
        if (bytesRead !== at.length) { index.delete(key); return null; }
        // A hit from an older generation is written again to the newest, so
        // rotation drops what has not been read lately: LRU by generation.
        if (at.gen !== current) enqueue(key, out);
        return out;
    };

    const rotate = async () => {
        await makeRootPrivate();
        await fs.mkdir(smallDir, { recursive: true, mode: DIR_MODE });
        const gens = await listGens();
        current = Math.max(current, gens.length ? gens[gens.length - 1] : 0) + 1;
        // Exclusive create; another process that rotated first owns the
        // name, and appending after its records is fine.
        await fs.writeFile(genPath(current), new Uint8Array(0), { flag: 'wx', mode: FILE_MODE }).catch((err) => {
            if (err?.code !== 'EEXIST') throw err;
        });
        for (const gen of (await listGens()).filter((g) => g <= current - GENERATIONS)) {
            await forgetGen(gen);
            await fs.rm(genPath(gen), { force: true });
        }
    };

    const flushSmall = async () => {
        while (queue.length) {
            const batch = queue;
            queue = [];
            const total = batch.reduce((n, r) => n + r.record.length, 0);
            if (!current || sealed) {
                await rotate();
                sealed = false;
            } else {
                const st = await fs.stat(genPath(current)).catch(() => null);
                if (!st || st.size + total > genBytes) await rotate();
            }
            const buf = Buffer.concat(batch.map((r) => r.record));
            const fh = await fs.open(genPath(current), 'a', FILE_MODE);
            let base;
            try {
                await fh.write(buf);
                // O_APPEND puts the batch at the end; its offset is the size
                // now minus its length. Another process appending in between
                // would make these offsets wrong, which costs only misses:
                // every hit is verified against its key.
                base = (await fh.stat()).size - buf.length;
            } finally {
                await fh.close();
            }
            let at = base;
            for (const r of batch) {
                index.set(r.key, { gen: current, offset: at + r.headerBytes, length: r.length });
                at += r.record.length;
            }
        }
    };

    const enqueue = (key, bytes) => {
        if (index.has(key) && index.get(key).gen === current) return;
        const raw = Buffer.from(key, 'hex');
        const headerBytes = 9 + raw.length;
        const record = Buffer.alloc(headerBytes + bytes.byteLength);
        record.writeUInt32LE(RECORD_MAGIC, 0);
        record[4] = raw.length;
        raw.copy(record, 5);
        record.writeUInt32LE(bytes.byteLength, 5 + raw.length);
        record.set(bytes, headerBytes);
        queue.push({ key, record, headerBytes, length: bytes.byteLength });
        // One append per burst: a query stores hundreds of rows at once.
        flushing ??= track(new Promise((resolve) => setImmediate(resolve))
            .then(() => flushSmall())
            .finally(() => { flushing = null; }));
        return flushing;
    };

    return {
        dir: root,
        async get(key) {
            checkKey(key);
            if (smallMaxBytes > 0) {
                await smallReady();
                const small = await readSmall(key);
                if (small) return small;
            }
            const file = fileOf(key);
            let bytes;
            try { bytes = await fs.readFile(file); } catch (err) {
                if (err?.code === 'ENOENT') return null;
                throw err;
            }
            // Touch for LRU; best effort.
            const now = new Date();
            fs.utimes(file, now, now).catch(() => {});
            return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        },
        put(key, bytes) {
            checkKey(key);
            if (bytes.byteLength < SMALL_ENTRY_BYTES) {
                if (smallMaxBytes === 0 || bytes.byteLength > genBytes) return Promise.resolve();
                return track(smallReady().then(() => enqueue(key, bytes)));
            }
            const file = fileOf(key);
            return track((async () => {
                if (bytes.byteLength > maxBytes) return;
                await makeRootPrivate();
                await fs.mkdir(path.dirname(file), { recursive: true, mode: DIR_MODE });
                // Write-then-rename, so a reader in another process sees
                // either no entry or a whole one.
                const tmp = `${file}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
                try {
                    await fs.writeFile(tmp, bytes, { mode: FILE_MODE });
                    await fs.rename(tmp, file);
                } catch (err) {
                    await fs.rm(tmp, { force: true });
                    throw err;
                }
                await evict(file);
            })());
        },
        async delete(key) {
            checkKey(key);
            // A log entry cannot be removed in place; forgetting it is enough,
            // and rotation reclaims the bytes.
            index.delete(key);
            await fs.rm(fileOf(key), { force: true });
        },
        // Settles every write started so far: for tests, and for a process
        // that wants its cache populated before it exits.
        async flush() {
            while (pending.size) await Promise.allSettled([...pending]);
        },
        // Closes the log's read handles (they otherwise live as long as the
        // process). The store stays usable; handles reopen on demand.
        async close() {
            await this.flush();
            for (const gen of [...handles.keys()]) {
                const handle = handles.get(gen);
                handles.delete(gen);
                await handle.then((h) => h.close(), () => {});
            }
        },
    };
}
