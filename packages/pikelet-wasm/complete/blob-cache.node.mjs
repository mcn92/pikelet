// A disk store for openPikeletFile's options.blobCache: whole regions a pack
// commits to by sha256 (the sketch's resident prefix, the query-interp
// segment holding the encoder), filed under that digest. Those regions are
// most of what a URL mount downloads (about 71 of 77 MiB on the 649 MiB
// Simple English Wikipedia pack), so a second process mounting the same pack
// reads them from disk instead of the network.
//
// The store is untrusted by design: the reader hashes every hit against the
// digest it asked for and treats a mismatch as a miss (blobCacheGet in
// pikelet-artifact-common.js), so a torn write or a corrupted file costs one
// refetch, never a wrong byte. Entries are immutable and never go stale; the
// only policy is a size budget, enforced by evicting least recently used
// entries (by mtime, touched on every hit) after each write.

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const KEY = /^[0-9a-f]{64}$/;
export const DEFAULT_BLOB_CACHE_BYTES = 2 * 1024 * 1024 * 1024;

// $PIKELET_CACHE_DIR, else $XDG_CACHE_HOME/pikelet/blobs, else
// ~/.cache/pikelet/blobs: the same root the CLI already caches encoder
// weights under.
export function defaultBlobCacheDir(env = process.env) {
    if (env.PIKELET_CACHE_DIR) return path.resolve(env.PIKELET_CACHE_DIR, 'blobs');
    const base = env.XDG_CACHE_HOME ? path.resolve(env.XDG_CACHE_HOME) : path.join(os.homedir(), '.cache');
    return path.join(base, 'pikelet', 'blobs');
}

export function diskBlobCache(dir = defaultBlobCacheDir(), { maxBytes = DEFAULT_BLOB_CACHE_BYTES } = {}) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
        throw new Error('diskBlobCache maxBytes must be a non-negative integer');
    }
    const root = path.resolve(dir);
    const fileOf = (key) => {
        if (!KEY.test(key)) throw new Error(`blob cache key must be a lowercase sha256 hex digest, got ${JSON.stringify(key)}`);
        return path.join(root, key.slice(0, 2), key);
    };
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

    return {
        dir: root,
        async get(key) {
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
            const file = fileOf(key);
            return track((async () => {
                if (bytes.byteLength > maxBytes) return;
                await fs.mkdir(path.dirname(file), { recursive: true });
                // Write-then-rename, so a reader in another process sees
                // either no entry or a whole one.
                const tmp = `${file}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
                try {
                    await fs.writeFile(tmp, bytes);
                    await fs.rename(tmp, file);
                } catch (err) {
                    await fs.rm(tmp, { force: true });
                    throw err;
                }
                await evict(file);
            })());
        },
        async delete(key) {
            await fs.rm(fileOf(key), { force: true });
        },
        // Settles every write started so far: for tests, and for a process
        // that wants its cache populated before it exits.
        async flush() {
            while (pending.size) await Promise.allSettled([...pending]);
        },
    };
}
