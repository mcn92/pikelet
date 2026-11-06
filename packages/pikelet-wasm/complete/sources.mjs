import { pikeletError, PIKELET_ERROR_CODES } from '../src/errors/pikelet-errors.js';

export function httpRangeSource(url, options = {}) {
    const stats = { requests: 0, bytes: 0, acceptRanges: null, etag: null, fullFallback: false, retries: 0, redirects: 0 };
    const maxFullFallbackBytes = options.maxFullFallbackBytes ?? 64 * 1024 * 1024;
    // Transient statuses are retried with capped exponential backoff
    // (Retry-After honored when sane): CDNs rate-limit the parallel range
    // bursts this reader legitimately issues — GitHub release assets answer
    // 429 under an open's prefetch + first-query fan-out — and failing a
    // whole query over a pressure signal wastes everything already fetched.
    const maxRetries = options.maxRetries ?? 6;
    const RETRYABLE = new Set([429, 502, 503, 504]);
    const retryDelayMs = (attempt, response) => {
        // GitHub's release CDN answers rate-limited range bursts with 429
        // and a Retry-After of up to a minute; honoring it (capped) beats
        // guessing. Without one, exponential backoff capped at 15 s.
        const after = Number(response.headers.get('retry-after'));
        if (Number.isFinite(after) && after >= 0 && after <= 90) return after * 1000;
        return Math.min(15000, 250 * 2 ** attempt);
    };
    // The transport is injectable so a runtime whose fetch decodes response
    // bodies can hand in one that does not (the Node entry, index.node.mjs,
    // defaults to nodeFetch, which never decompresses). The default here is
    // looked up per call, as a bare fetch() would be.
    const fetch = typeof options.fetch === 'function' ? options.fetch : (input, init) => globalThis.fetch(input, init);
    const cacheKeyParam = options.cacheKeyParam === undefined ? 'r'
        : (options.cacheKeyParam === null || options.cacheKeyParam === false) ? null
            : String(options.cacheKeyParam);
    let full = null;
    // A parent reached through a publisher-supplied locator is confined to the
    // child's origin and directory, and 5.1.1 applies that rule to EVERY
    // redirect hop before it is followed. The guard throws to refuse a hop.
    // With a guard set, reads never auto-follow: only the HEAD walk below,
    // which checks each hop, may move the target.
    const redirectGuard = typeof options.redirectGuard === 'function' ? options.redirectGuard : null;
    const readInit = (headers) => (redirectGuard ? { headers, redirect: 'error' } : { headers });

    // Redirect memoization. A host that answers with a redirect (GitHub
    // release assets: github.com 302s every hit to a signed CDN URL) would
    // otherwise charge its rate-limited front door once per range read —
    // ~130 hits for one wiki query, and github.com's frontend throttling,
    // not the CDN, is what 429s the burst; the blob host behind the
    // redirect serves parallel 206s happily. Resolve the chain once, pin
    // the final URL, and re-resolve only when it stops being accepted
    // (401/403 — signed URLs expire). Browsers cannot read Location from a
    // cross-origin redirect (opaqueredirect), so resolution degrades to
    // the un-pinned auto-follow behavior there.
    let resolvedTarget = null; // { target, viaRedirect } | null (unresolved or unresolvable)
    let resolving = null;
    let sizeFromHead = null;
    let pinnedOnce = false;
    const resolveTarget = async () => {
        if (resolvedTarget) return resolvedTarget;
        resolving ??= (async () => {
            let target = url;
            let viaRedirect = false;
            try {
                for (let hop = 0; hop < 5; hop++) {
                    const probe = await fetch(target, { method: 'HEAD', redirect: 'manual' });
                    const location = probe.headers.get('location');
                    if (probe.status >= 300 && probe.status < 400 && location) {
                        const next = new URL(location, target).href;
                        if (new URL(target).protocol === 'https:' && !next.startsWith('https:')) {
                            throw Object.assign(new Error(`refused redirect from https to ${next}`), { redirectRefused: true });
                        }
                        if (redirectGuard) {
                            try { redirectGuard(next); } catch (err) {
                                throw Object.assign(new Error(`refused redirect to ${next}: ${err.message}`), { redirectRefused: true });
                            }
                        }
                        target = next;
                        viaRedirect = true;
                        stats.redirects += 1;
                        continue;
                    }
                    if (probe.type === 'opaqueredirect' || probe.status === 0) {
                        // Browser CORS: the chain is invisible; let fetch
                        // auto-follow per read.
                        target = url;
                        viaRedirect = false;
                    } else if (probe.ok) {
                        const len = Number(probe.headers.get('content-length'));
                        const etag = probe.headers.get('etag');
                        // Re-resolution (a signed URL expired) must land on
                        // the artifact first pinned. Adopting the new HEAD's
                        // ETag would make the If-Range / ARTIFACT_CHANGED
                        // check compare a replaced file against itself.
                        if (pinnedOnce) {
                            const sizeChanged = sizeFromHead !== null && Number.isFinite(len) && len > 0 && len !== sizeFromHead;
                            if ((stats.etag && etag && etag !== stats.etag) || sizeChanged) {
                                throw Object.assign(pikeletError(
                                    PIKELET_ERROR_CODES.ARTIFACT_CHANGED,
                                    `artifact changed on the host: expected ETag ${stats.etag || '(none)'}, size ${sizeFromHead ?? '(unknown)'}; `
                                        + `re-resolved ${etag || '(none)'}, size ${Number.isFinite(len) ? len : '(unknown)'}`,
                                    { expectedEtag: stats.etag, actualEtag: etag },
                                ), { artifactChanged: true });
                            }
                        } else {
                            if (Number.isFinite(len) && len > 0) sizeFromHead = len;
                            stats.acceptRanges = probe.headers.get('accept-ranges');
                            stats.etag = etag;
                        }
                    }
                    break;
                }
            } catch (err) {
                // A refused hop is a verdict, not a probe failure: falling back
                // to the original URL would retry the same redirect. So is a
                // replaced artifact.
                if (err?.redirectRefused || err?.artifactChanged) {
                    resolving = null;
                    throw err;
                }
                // HEAD unsupported or blocked: fall back to auto-follow.
                target = url;
                viaRedirect = false;
            }
            resolvedTarget = { target, viaRedirect };
            pinnedOnce = true;
            resolving = null;
            return resolvedTarget;
        })();
        return resolving;
    };

    return {
        stats,
        // The location as given, before any redirect: what a child's parent
        // locator resolves against (5.1.1).
        url,
        // The transport, when one was injected: a chain walk opens the
        // parents it reaches over HTTP with the same one (layer-reader.mjs).
        fetch: typeof options.fetch === 'function' ? options.fetch : undefined,
        size: undefined,
        preferredParallelism: options.preferredParallelism ?? 32,
        // Rows closer than this are fetched in one request, reading the
        // bytes between them. Over HTTP a round trip costs far more than
        // the bytes: on the 648.5 MB wiki pack in Chromium, 256 KiB took
        // 61 -> 38 requests and about 2.8 -> 2.2 s per query against 16 KiB,
        // at 0.9 -> 2.3 MiB transferred.
        preferredGapBytes: options.preferredGapBytes ?? 256 * 1024,

        async init() {
            await resolveTarget();
            if (sizeFromHead !== null) this.size = sizeFromHead;
        },

        async read(offset, length) {
            // After a one-time full download (host ignored Range) every read
            // is served from memory; no further requests go out.
            if (full) return full.subarray(offset, offset + length);
            stats.requests += 1;
            stats.bytes += length;
            const end = offset + length - 1;
            const headers = { Range: `bytes=${offset}-${end}` };
            // RFC 9110: If-Range takes a strong validator only; a strict host
            // ignores Range when given a weak one, forcing the full download.
            if (stats.etag && !stats.etag.startsWith('W/')) headers['If-Range'] = stats.etag;
            const pinned = await resolveTarget();
            // The per-range cache-key query defeats Chromium's same-URL
            // cache-entry lock (concurrent fetches of one cacheable URL
            // serialize on it). Hosts that sign the full query string
            // (S3 presigned URLs and similar) can turn it off with
            // options.cacheKeyParam: null — range reads then share the
            // unmodified URL. A redirect-pinned target never gets the
            // param: signed URLs sign their query string, and appending
            // anything invalidates them.
            const targetFor = ({ target, viaRedirect }) => (cacheKeyParam && !viaRedirect
                ? `${target}${target.includes('?') ? '&' : '?'}${cacheKeyParam}=${offset}-${end}`
                : target);
            let response = await fetch(targetFor(pinned), readInit(headers));
            if ((response.status === 401 || response.status === 403) && pinned.viaRedirect) {
                // The pinned signed URL expired; resolve a fresh one once
                // and retry. Concurrent reads share the re-resolution.
                if (resolvedTarget === pinned) resolvedTarget = null;
                stats.retries += 1;
                response = await fetch(targetFor(await resolveTarget()), readInit(headers));
            }
            for (let attempt = 0; RETRYABLE.has(response.status) && attempt < maxRetries; attempt++) {
                stats.retries += 1;
                await new Promise((resolve) => setTimeout(resolve, retryDelayMs(attempt, response)));
                response = await fetch(targetFor(resolvedTarget || pinned), readInit(headers));
            }

            // A pack is content-addressed bytes: a body the host encoded is
            // not those bytes, and an encoded range cannot be sliced anyway.
            // Fail closed wherever the header is visible (a browser hides it
            // on cross-origin responses unless the host exposes it).
            const encoding = response.headers.get('content-encoding');
            const codings = String(encoding || '').split(',').map((c) => c.trim().toLowerCase()).filter((c) => c && c !== 'identity');
            if ((response.status === 206 || response.status === 200) && codings.length) {
                await response.body?.cancel?.().catch(() => {});
                throw new Error(`range read returned Content-Encoding: ${encoding}; a .pikelet must be served unencoded`);
            }
            if (response.status === 206) {
                // A 206 must be the range we asked for and say so (RFC 9110 requires
                // Content-Range): a wrong slice would otherwise be caught only by a
                // downstream digest, and format-1 sketch rows have no per-row digest.
                const contentRange = response.headers.get('content-range');
                const m = contentRange && /^bytes (\d+)-(\d+)\//.exec(contentRange);
                if (!m || Number(m[1]) !== offset || Number(m[2]) > end) {
                    throw new Error(`range read returned Content-Range ${contentRange || '(missing)'}, requested bytes ${offset}-${end}`);
                }
                const body = new Uint8Array(await response.arrayBuffer());
                if (body.length !== Number(m[2]) - Number(m[1]) + 1) {
                    throw new Error(`range read body is ${body.length} bytes for Content-Range ${contentRange}`);
                }
                return body;
            }
            if (response.status === 200) {
                // Under If-Range, a 200 instead of 206 means the host ignores Range or the
                // pinned ETag no longer matches (the artifact was replaced). Only the first
                // may fall back to a full download: slices of a replaced file don't match
                // the manifest, and a large lexical segment has only a whole-segment digest.
                // So fall back only when the ETag still matches (or neither side has one).
                if (stats.etag) {
                    const responseEtag = response.headers.get('etag');
                    if (responseEtag !== stats.etag) {
                        throw pikeletError(
                            PIKELET_ERROR_CODES.ARTIFACT_CHANGED,
                            `artifact changed on the host: expected ETag ${stats.etag}, got ${responseEtag || '(missing)'}`,
                            { expectedEtag: stats.etag, actualEtag: responseEtag },
                        );
                    }
                }
                // An unknown size must refuse, not pass the gate: with no
                // Content-Length (chunked body) and init() never called,
                // size would be 0 and the cap check silently vacuous.
                const size = Number(response.headers.get('content-length')) || this.size || 0;
                if (size <= 0) {
                    throw new Error('host ignores Range and reports no size; refusing unbounded full download (call init() first or fix the host)');
                }
                if (size > maxFullFallbackBytes) {
                    throw new Error(`host ignores Range and the file is ${size} bytes; refusing full download`);
                }
                // Declared sizes are advisory (a chunked body can stream more
                // than any header claimed), so the cap is enforced on the
                // bytes actually received, aborting as soon as it is crossed.
                const reader = response.body?.getReader?.();
                if (!reader) {
                    const bytes = new Uint8Array(await response.arrayBuffer());
                    if (bytes.length > maxFullFallbackBytes) {
                        throw new Error(`host ignores Range and streamed ${bytes.length} bytes; refusing full download over ${maxFullFallbackBytes}`);
                    }
                    full = bytes;
                } else {
                    const chunks = [];
                    let received = 0;
                    for (;;) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        received += value.byteLength;
                        if (received > maxFullFallbackBytes) {
                            await reader.cancel().catch(() => {});
                            throw new Error(`host ignores Range and streamed over ${maxFullFallbackBytes} bytes; aborting full download`);
                        }
                        chunks.push(value);
                    }
                    full = new Uint8Array(received);
                    let at = 0;
                    for (const chunk of chunks) { full.set(chunk, at); at += chunk.byteLength; }
                }
                stats.fullFallback = true;
                console.warn('.pikelet host does not honor Range; fell back to a one-time full download');
                return full.subarray(offset, offset + length);
            }
            throw new Error(`range read failed: HTTP ${response.status}`);
        },
    };
}
