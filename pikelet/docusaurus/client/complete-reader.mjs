// Thin adapter over the packaged one-file reader (pikelet-wasm/complete).
// The primary path range-reads the artifact over HTTP 206 — the sketch
// profile's execution model, with the reader's own graceful fallback to a
// bounded full download when a host ignores Range (the Cloudflare Pages
// lesson) — and defers the kind-3 encoder region to a background prefetch
// so the panel is interactive on a fraction of the file. The memory-bytes
// variant stays for callers that already hold the artifact. The plugin's
// webpack alias points the bare specifier at the site-resolved module (the
// repo root when building inside the pikelet monorepo), exactly like
// pikelet-wasm/artifact. This file used to be a second implementation of
// the PSF1 reader; keeping it as an adapter preserves the widget's API
// while the format logic lives in one place.
import { openPikeletFile, httpRangeSource } from 'pikelet-wasm/complete';

export async function openCompletePikeletUrl(url, options = {}) {
  const source = httpRangeSource(url, {
    // Bound the ignores-Range fallback well above docs-scale artifacts;
    // wiki-scale files should fail loudly rather than pull 600 MiB.
    maxFullFallbackBytes: options.maxFullFallbackBytes ?? 96 * 1024 * 1024,
  });
  await source.init();
  const search = await openPikeletFile(source, {
    rerankParallelism: options.rerankParallelism,
    rerankGap: options.rerankGap,
    prefetchEncoder: options.prefetchEncoder,
  });
  return {
    info: () => search.info(),
    stats: () => ({ ...source.stats }),
    query(text, queryOptions = {}) {
      return search.query(text, { k: 8, rerank: options.rerank, ...queryOptions });
    },
    close: () => search.close(),
  };
}

function memorySource(bytes) {
  return {
    size: bytes.length,
    preferredParallelism: Infinity,
    preferredGapBytes: 2048,
    async read(offset, length) {
      return bytes.subarray(offset, offset + length);
    },
    async close() {},
  };
}

export async function openCompletePikelet(bytes, options = {}) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const search = await openPikeletFile(memorySource(u8), {
    rerankParallelism: options.rerankParallelism,
    rerankGap: options.rerankGap,
  });
  return {
    info: () => search.info(),
    query(text, queryOptions = {}) {
      return search.query(text, { k: 8, rerank: options.rerank, ...queryOptions });
    },
    close: () => search.close(),
  };
}
