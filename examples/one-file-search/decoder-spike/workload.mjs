// The browser test's workload (browser_check.mjs), run unchanged by Node and
// by each browser: no Node or DOM APIs. For each config it hashes every
// logits row the kernel writes (FNV-1a over the f32 bits), so two runs agree
// only if every logit is bit-identical, and times decode and prefill.
//   decode:  the first N long-text tokens, one per call (crosses the 256 window)
//   prefill: the same tokens in 200-token calls, every row's logits
//   greedy:  each prompt, then 100 greedy tokens
const V = 50257, EOS = 50256;
export const CONFIGS = [
  { name: 'u8', blob: 'tinystories-u8.bin', bits: 8 },
  { name: 'u4', blob: 'tinystories-u4.bin', bits: 4 },
  { name: 'u5', blob: 'tinystories-u5.bin', bits: 5 },
  { name: 'u5 expanded', blob: 'tinystories-u5.bin', bits: 6 },
  { name: 'u8 shortlist r192 K256', blob: 'tinystories-u8.bin', bits: 8, shortlist: 'shortlist-r192.bin', r: 192, k: 256 },
];

function fnv(h, u32) {
  for (let i = 0; i < u32.length; i++) {
    const x = u32[i];
    h = Math.imul(h ^ (x & 0xff), 16777619);
    h = Math.imul(h ^ ((x >>> 8) & 0xff), 16777619);
    h = Math.imul(h ^ ((x >>> 16) & 0xff), 16777619);
    h = Math.imul(h ^ (x >>> 24), 16777619);
  }
  return h >>> 0;
}

export async function runWorkload(mod, load, { longIds, prompts, nLong = 600, nGen = 100, configs = null }, log = () => {}) {
  const results = [];
  for (const c of CONFIGS.filter((x) => !configs || configs.includes(x.name))) {
    log(`${c.name}: loading`);
    const bytes = await load(c.blob);
    let blob = mod._malloc(bytes.length);
    mod.HEAPU8.set(bytes, blob);
    let raw = 0, sl = 0;
    if (c.bits === 6) {
      raw = blob;
      blob = mod._malloc(Number(mod._decoder_expand_u5(raw, 0)));
      mod._decoder_expand_u5(raw, blob);
    }
    if (c.shortlist) {
      const s = await load(c.shortlist);
      sl = mod._malloc(s.length);
      mod.HEAPU8.set(s, sl);
      if (mod._decoder_set_shortlist(sl, c.r, c.k)) throw new Error('set_shortlist');
    }
    const idsPtr = mod._malloc(4 * 256), lg = mod._malloc(4 * V * 200);
    const fwd = (ids, pos0, all) => {
      mod.HEAP32.set(ids, idsPtr >> 2);
      if (mod._decoder_forward(blob, idsPtr, ids.length, pos0, lg, c.bits, all ? 1 : 0)) throw new Error('forward');
      return new Uint32Array(mod.HEAPU8.buffer, lg, V * (all ? ids.length : 1));
    };
    const argmax = () => {
      const r = new Float32Array(mod.HEAPU8.buffer, lg, V);
      let b = 0; for (let i = 1; i < V; i++) if (r[i] > r[b]) b = i;
      return b;
    };
    const ids = longIds.slice(0, nLong);
    const out = { name: c.name };

    let t = performance.now();
    out.decode = ids.map((id, p) => fnv(2166136261, fwd([id], p, false)));
    out.decodeTps = nLong / (performance.now() - t) * 1000;
    log(`${c.name}: decode ${out.decodeTps.toFixed(0)} tok/s`);

    if (!c.shortlist) {  // the shortlist only applies to last-row logits
      t = performance.now();
      out.prefill = [];
      for (let p0 = 0; p0 < nLong; p0 += 200) {
        const chunk = ids.slice(p0, p0 + 200), rows = fwd(chunk, p0, true);
        for (let r = 0; r < chunk.length; r++) out.prefill.push(fnv(2166136261, rows.subarray(r * V, (r + 1) * V)));
      }
      out.prefillTps = nLong / (performance.now() - t) * 1000;
      log(`${c.name}: prefill ${out.prefillTps.toFixed(0)} tok/s`);
    }

    let genMs = 0, genTok = 0;
    out.greedy = prompts.map((p) => {
      fwd(p, 0, false);
      const g = [];
      let next = argmax(), pos = p.length;
      const t0 = performance.now();
      while (g.length < nGen && next !== EOS) { g.push(next); fwd([next], pos++, false); next = argmax(); }
      genMs += performance.now() - t0; genTok += g.length;
      return g;
    });
    out.genTps = genTok / genMs * 1000;
    log(`${c.name}: greedy ${out.genTps.toFixed(0)} tok/s`);

    if (sl) { mod._decoder_set_shortlist(0, 0, 0); mod._free(sl); }
    mod._free(lg); mod._free(idsPtr); mod._free(blob);
    if (raw) mod._free(raw);
    results.push(out);
  }
  return results;
}
