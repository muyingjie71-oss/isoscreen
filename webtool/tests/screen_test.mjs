// Full browser pipeline in Node (RDKit.js featuriser + JS solute encoders + onnxruntime-web WASM heads, float16 assets)
// vs Python production predictors on the 5,040-point grid; then timing of one full screen (2 solutes x 6,911 solvents), single thread.
import fs from 'fs'; import { createRequire } from 'module'; const require = createRequire(import.meta.url);
const ort = require('onnxruntime-web'); ort.env.wasm.numThreads = 1;
const R = await require('@rdkit/rdkit')(); const { makeFeaturizer } = await import('../site/js/featurize.js');
const E = await import('../site/js/engine.js'); const { SliceScreener, f16ToF32 } = await import('../site/js/screen.js');
const A = new URL('../build/work/assets/', import.meta.url).pathname; const rd = (p) => fs.readFileSync(A + p);   // assets written by webtool/build
const CFG = { onnx: process.env.ONNX || 'onnx32', jsw: process.env.JSW || 'mix', emb: process.env.EMB || 'f16' };   // the precision used by the site
const meta = JSON.parse(rd('weights_js.json'));
const w16 = f16ToF32(new Uint16Array(rd('weights_js.f16').buffer.slice(0))), w32 = new Float32Array(rd('weights_js.f32').buffer.slice(0));
let wbuf = CFG.jsw === 'f32' ? w32 : w16;
if (CFG.jsw === 'mix') { wbuf = Float32Array.from(w32); const [o, sh] = [0, 0]; for (const [nm, [off, shape]] of Object.entries(meta.index)) if (nm.includes('enet2/W')) { const n = shape.reduce((a, b) => a * b, 1); wbuf.set(w16.subarray(off, off + n), off); } }
const W = E.loadWeights(meta, wbuf); const seeds = meta.seeds, nk = seeds.length;
const solv = JSON.parse(rd('solvents.json')); const S = solv.length;
const ld = (n) => CFG.emb === 'f32' ? new Float32Array(rd(n + '.f32').buffer.slice(0)) : f16ToF32(new Uint16Array(rd(n + '.f16').buffer.slice(0))); const dmAll = ld('dmpnn_solvents'), ghAll = ld('ghgnn_solvents');
const onnx = {}; for (const f of fs.readdirSync(A + CFG.onnx)) onnx[f.replace('.onnx', '')] = new Uint8Array(rd(CFG.onnx + '/' + f));
const enet = seeds.map((_, k) => ({ w1: W[`g${k}/enet1/W`].d, b1: W[`g${k}/enet1/b`].d, W2: W[`g${k}/enet2/W`].d, b2: W[`g${k}/enet2/b`].d }));
const F = makeFeaturizer(R, JSON.parse(fs.readFileSync('../site/js/elements.json')));
const enc = (smi) => { const P = F.parse(smi); const dg = F.dmpnnGraph(P); let gg = null; try { gg = F.ghgnnGraph(P); } catch (e) {}
  return { dm: seeds.map((_, k) => Float32Array.from(E.dmpnnEncode(W, k, dg))), gh: gg ? seeds.map((_, k) => Float32Array.from(E.ghgnnEncode(W, k, gg, meta.ghgnn.eps))) : null, hba: gg?.hba ?? 0, hbd: gg?.hbd ?? 0 }; };
function sliceOf(idx) {
  const n = idx.length, dm = new Float32Array(nk * n * 256), gh = new Float32Array(nk * n * 226);
  for (let k = 0; k < nk; k++) idx.forEach((si, j) => { dm.set(dmAll.subarray((k * S + si) * 256, (k * S + si + 1) * 256), (k * n + j) * 256); gh.set(ghAll.subarray((k * S + si) * 226, (k * S + si + 1) * 226), (k * n + j) * 226); });
  return { lo: 0, hi: n, dm, gh, hba: idx.map((i) => solv[i].hba), hbd: idx.map((i) => solv[i].hbd) };
}
const csv = (p) => { const [h, ...r] = fs.readFileSync(p, 'utf8').trim().split(/\r?\n/); const c = h.split(','); return r.map((l) => Object.fromEntries(l.split(',').map((v, i) => [c[i], v]))); };
const mol = Object.fromEntries(csv(new URL('../../training/inputs/molecules_as_trained.csv', import.meta.url).pathname).map((r) => [Object.values(r)[0], r.SMILES]));
const refD = csv(new URL('../../models/acceptance/grid_dmpnn_cpu.csv', import.meta.url).pathname), refG = csv(new URL('../../models/acceptance/grid_ghgnn_cpu.csv', import.meta.url).pathname);
const key = (r) => `${r.solute_id}|${r.solvent_id}|${r.T_K}`; const mG = new Map(refG.map((r) => [key(r), r]));
const solvIds = [...new Set(refD.map((r) => r.solvent_id))], soluIds = [...new Set(refD.map((r) => r.solute_id))];
const scr = await SliceScreener.create(ort, onnx, meta, sliceOf(solvIds.map((id) => solv.findIndex((s) => s.solvent_id === id))), enet);
const Ts = [298.15, 323.15, 373.15]; const st = { d: 0, g: 0, c: 0, dS: 0 }; const dsel = [];
const cache = {};
for (const uid of soluIds) {
  const [o] = await scr.run([enc(mol[uid])], Ts); cache[uid] = o;
  for (const r of refD.filter((x) => x.solute_id === uid)) {
    const j = solvIds.indexOf(r.solvent_id), t = Ts.indexOf(+r.T_K); let md = 0, mg = 0;
    for (let k = 0; k < nk; k++) { md += o.dm[(k * scr.n + j) * 3 + t] / nk; mg += (o.A[k * scr.n + j] + o.B[k * scr.n + j] / Ts[t]) / nk; }
    const g = mG.get(key(r)); st.d = Math.max(st.d, Math.abs(md - +r.pred_mean)); if (!Number.isNaN(mg)) { st.g = Math.max(st.g, Math.abs(mg - +g.pred_mean)); st.c = Math.max(st.c, Math.abs((md + mg) / 2 - (+r.pred_mean + +g.pred_mean) / 2)); }
  }
}
console.log(`${JSON.stringify(CFG)} vs Python, max |diff ln g|: D-MPNN ${st.d.toExponential(2)}, GH-GNN ${st.g.toExponential(2)}, consensus ${st.c.toExponential(2)}`);
if (process.env.FAST) process.exit(0);
// timing on the full library, single thread
let t0 = performance.now(); const full = await SliceScreener.create(ort, onnx, meta, sliceOf([...Array(S).keys()]), enet); const tP = performance.now() - t0;
const sol = [enc('Cc1ccccc1C'), enc('Cc1ccc(C)cc1')];
t0 = performance.now(); const out = await full.run(sol, [323.15]); const tR = performance.now() - t0;
t0 = performance.now(); await full.run(sol, [348.15], false); const tT = performance.now() - t0;
console.log(`single thread: load+prepare ${(tP / 1000).toFixed(1)} s, full screen ${(tR / 1000).toFixed(1)} s, new temperature only ${(tT / 1000).toFixed(1)} s`);
fs.writeFileSync('/tmp/oxy_pxy_browser.json', JSON.stringify({ dm0: Array.from(out[0].dm.slice(0, 5)), A0: Array.from(out[0].A.slice(0, 5)) }));
