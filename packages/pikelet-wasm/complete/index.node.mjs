// The Node entry of 'pikelet-wasm/complete': the portable reader, except
// that httpRangeSource defaults to nodeFetch (node:http/node:https, no
// automatic decompression, Content-Encoding other than identity refused)
// instead of the runtime's fetch, which decodes whatever a server labels as
// encoded. Pass options.fetch to choose otherwise.

import { httpRangeSource as portableHttpRangeSource } from './sources.mjs';
import { nodeFetch } from './node-fetch.mjs';

export * from './index.mjs';
export { nodeFetch };
export { diskBlobCache, defaultBlobCacheDir, DEFAULT_BLOB_CACHE_BYTES, DEFAULT_SMALL_CACHE_BYTES } from './blob-cache.node.mjs';

export function httpRangeSource(url, options = {}) {
    return portableHttpRangeSource(url, { fetch: nodeFetch, ...options });
}
