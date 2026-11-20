// GPT-2 byte-level BPE (vocab.json + merges.txt), encode and decode. For
// SmolLM2-style tokenizers (llama-tokenizer/<name>.json from export_llama.py),
// createBpeFromJson adds special tokens (matched literally before BPE) and
// the Digits pre-tokenizer (every digit its own piece).
const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function byteTable() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const enc = new Array(256);
  bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); });
  return enc;
}

export function createBpeFromJson({ vocab, merges, specials = {}, individualDigits = false }) {
  return createBpe(vocab, merges, { specials, individualDigits });
}

export function createBpe(vocab, merges, { specials = {}, individualDigits = false } = {}) {
  const ranks = new Map();
  const mergeList = Array.isArray(merges) ? merges : merges.split('\n').filter((l) => l && !l.startsWith('#version'));
  mergeList.forEach((l, i) => ranks.set(l, i));
  const specialNames = Object.keys(specials).sort((a, b) => b.length - a.length);
  const specialRe = specialNames.length
    ? new RegExp(`(${specialNames.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`)
    : null;
  const byteEnc = byteTable();
  const byteDec = new Map(byteEnc.map((c, b) => [c, b]));
  const idToTok = [];
  for (const [t, id] of Object.entries(vocab)) idToTok[id] = t;
  const specialById = new Map(Object.entries(specials).map(([t, id]) => [id, t]));
  const cache = new Map();
  const utf8 = new TextEncoder();

  function bpe(word) {
    if (cache.has(word)) return cache.get(word);
    let parts = [...word];
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = ranks.get(parts[i] + ' ' + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1];
      const next = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { next.push(a + b); i++; }
        else next.push(parts[i]);
      }
      parts = next;
    }
    cache.set(word, parts);
    return parts;
  }

  return {
    encode(text) {
      const ids = [];
      const pieces = specialRe ? text.split(specialRe) : [text];
      for (const piece of pieces) {
        if (!piece) continue;
        if (specialRe && specials[piece] !== undefined) { ids.push(specials[piece]); continue; }
        for (const seg of individualDigits ? piece.split(/(\p{N})/u) : [piece]) {
          if (!seg) continue;
          for (const m of seg.matchAll(PAT)) {
            const word = Array.from(utf8.encode(m[0]), (b) => byteEnc[b]).join('');
            for (const p of bpe(word)) ids.push(vocab[p]);
          }
        }
      }
      return ids;
    },
    // Bytes, not a string: a token can end mid-character.
    decodeBytes(ids) {
      const out = [];
      for (const id of ids) {
        if (specialById.has(id)) { for (const b of utf8.encode(specialById.get(id))) out.push(b); continue; }
        for (const ch of idToTok[id]) out.push(byteDec.get(ch));
      }
      return Uint8Array.from(out);
    },
  };
}
