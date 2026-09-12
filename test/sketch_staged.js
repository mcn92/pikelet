// Staged-boot micro tier conformance: format compatibility, stage-1
// integrity, tier semantics, and convergence to full-tier determinism.
const fs = require('fs');
const os = require('os');
const path = require('path');
const Pikelet = require('../pikelet.js');
const { exportSketchArtifact } = require('../pikelet-artifact.js');

let passed = 0, failed = 0;
function check(label, cond) {
  if (cond) { passed++; console.log('  ok:', label); }
  else { failed++; console.log('  FAIL:', label); }
}

function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

async function main() {
  const dim = 64, count = 3000, K = 10;
  const rand = mulberry32(1234);
  // unit-norm rows, per-row affine u8 quantization (spec 2.2)
  const qdata = new Uint8Array(count * dim);
  const scales = new Float32Array(count), offsets = new Float32Array(count);
  const raw = new Float32Array(count * dim);
  for (let i = 0; i < count; i++) {
    let n = 0;
    for (let d = 0; d < dim; d++) { const v = rand() * 2 - 1; raw[i*dim+d] = v; n += v*v; }
    n = 1/Math.sqrt(n);
    let mn = Infinity, mx = -Infinity;
    for (let d = 0; d < dim; d++) { const v = raw[i*dim+d]*n; raw[i*dim+d] = v; if (v<mn) mn=v; if (v>mx) mx=v; }
    const s = (mx-mn)/255 || 1e-12;
    scales[i]=s; offsets[i]=mn;
    for (let d = 0; d < dim; d++) {
      let b = Math.round((raw[i*dim+d]-mn)/s);
      qdata[i*dim+d] = b<0?0:b>255?255:b;
    }
  }
  const idx = { dim, count, metric: 1, qdata, scales, offsets };
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-staged-'));
  const plainPath = path.join(tmp, 'plain.pikelet-sketch');
  const microPath = path.join(tmp, 'micro.pikelet-sketch');
  const opts = { sketchDims: 32, sketchBits: 8, recommendedRerank: 60 };
  exportSketchArtifact(idx, plainPath, opts);
  const info = exportSketchArtifact(idx, microPath, { ...opts, microDims: 16, microBits: 8 });
  check('builder reports micro tier + stage-1 bytes', !!info.micro && info.micro.stage1Bytes < info.sketch.residentBytes);

  const queries = [];
  for (let q = 0; q < 12; q++) {
    const v = new Float32Array(dim);
    let n = 0;
    for (let d = 0; d < dim; d++) { const x = rand()*2-1; v[d]=x; n+=x*x; }
    n = 1/Math.sqrt(n);
    for (let d = 0; d < dim; d++) v[d]*=n;
    queries.push(v);
  }

  // 1. v1-reader compatibility: non-staged open of a micro file must equal
  // a plain file built from the same rows, byte-for-byte on results.
  const plain = await Pikelet.SketchArtifact.openFile(plainPath);
  const microPlainOpen = await Pikelet.SketchArtifact.openFile(microPath);
  let same = true;
  for (const q of queries) {
    const a = await plain.search(q, K, { rerank: 60 });
    const b = await microPlainOpen.search(q, K, { rerank: 60 });
    if (JSON.stringify(a.results) !== JSON.stringify(b.results) || b.tier !== 'full') same = false;
  }
  check('non-staged open of micro file matches plain artifact exactly', same);

  // 2. staged open: micro tier first, boosted C, then convergence.
  const stages = [];
  const staged = await Pikelet.SketchArtifact.open(
    { read: (o, l) => Promise.resolve(new Uint8Array(fs.readFileSync(microPath).buffer.slice(o, o + l))) },
    { staged: true, onStage: (s) => stages.push(s.tier) });
  check('staged open starts in micro tier', staged.tier === 'micro');
  check('stage-1 resident bytes < full resident bytes', staged.residentBytes < info.sketch.residentBytes);
  const early = await staged.search(queries[0], K);
  check('micro-tier search reports tier and boosted rerank', early.tier === 'micro' && early.rerank === 60 * 4);
  await staged.fullyResident;
  check('convergence swaps to full tier', staged.tier === 'full' && stages.join(',') === 'micro,full');
  let conv = true;
  for (const q of queries) {
    const a = await plain.search(q, K, { rerank: 60 });
    const b = await staged.search(q, K, { rerank: 60 });
    if (JSON.stringify(a.results) !== JSON.stringify(b.results) || b.tier !== 'full') conv = false;
  }
  check('post-convergence results identical to non-staged open', conv);

  // 3. per-tier WASM scanners agree with the JS tier scan.
  const microScanner = await Pikelet.createSketchScanner(staged, { tier: 'micro', maxRerank: count });
  const fullScanner = await Pikelet.createSketchScanner(staged, { tier: 'full', maxRerank: count });
  check('scanner tags tier + dims', microScanner.tier === 'micro' && microScanner.sketchDims === 16 && fullScanner.sketchDims === 32);
  const viaFull = await staged.search(queries[1], K, { rerank: 60, scanner: fullScanner });
  const viaJs = await staged.search(queries[1], K, { rerank: 60 });
  check('full-tier WASM scan matches JS scan results', JSON.stringify(viaFull.results) === JSON.stringify(viaJs.results));
  const dimsMismatch = await staged.search(queries[2], K, { rerank: 60, scanner: microScanner });
  check('mismatched-tier scanner is bypassed, not misused', dimsMismatch.tier === 'full' && JSON.stringify(dimsMismatch.results) === JSON.stringify((await plain.search(queries[2], K, { rerank: 60 })).results));

  // 4. integrity: stage-1 tamper rejected at stage 1; full-sketch tamper
  // passes stage 1 but must fail the background completion.
  const microInfo = exportSketchArtifact(idx, microPath + '.t1', { ...opts, microDims: 16 });
  const t1 = fs.readFileSync(microPath + '.t1');
  t1[microInfo.addressing.sketchesOffset + count * 32 + 5] ^= 0xff; // micro segment
  fs.writeFileSync(microPath + '.t1', t1);
  let s1rej = false;
  try { await Pikelet.SketchArtifact.open({ read: (o,l)=>Promise.resolve(new Uint8Array(fs.readFileSync(microPath+'.t1').buffer.slice(o,o+l))) }, { staged: true }); }
  catch (e) { s1rej = true; }
  check('tampered micro segment rejected at stage 1', s1rej);

  const t2 = fs.readFileSync(microPath);
  t2[microInfo.addressing.sketchesOffset + 7] ^= 0xff; // full-sketch segment
  fs.writeFileSync(microPath + '.t2', t2);
  const lateTamper = await Pikelet.SketchArtifact.open({ read: (o,l)=>Promise.resolve(new Uint8Array(fs.readFileSync(microPath+'.t2').buffer.slice(o,o+l))) }, { staged: true });
  check('full-sketch tamper is invisible to stage 1', lateTamper.tier === 'micro');
  let s2rej = false;
  try { await lateTamper.fullyResident; } catch (e) { s2rej = true; }
  check('full-sketch tamper rejected at stage 2', s2rej && lateTamper.tier === 'micro');

  console.log(`Staged sketch conformance: ${passed} passed, ${failed} failed`);
  if (failed) process.exit(1);
}
main().catch((e) => { console.error(e); process.exit(1); });
