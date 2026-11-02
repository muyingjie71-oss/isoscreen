// IsoScreen front end. All computation runs in the browser: RDKit.js reads the two isomers, the solute encoders run here,
// and the solvent library is screened in Web Workers with onnxruntime-web. Numbers on the Method page are read from data/validation.json.
import { makeFeaturizer } from './featurize.js';
import { loadWeights, dmpnnEncode, ghgnnEncode } from './engine.js';
import { f16ToF32 } from './screen.js';
import { makeIsoType } from './isotype.js';

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (x, d = 3) => (x == null || !Number.isFinite(x) ? '–' : x.toFixed(d));
const fmtCap = (x) => (!Number.isFinite(x) ? '–' : x >= 0.01 ? x.toFixed(3) : x.toExponential(1).replace('e', '×10<sup>').replace(/$/, '</sup>').replace('<sup>-', '<sup>−'));
const pct = (x) => `${Math.round(100 * x)}%`;
const LEAST = new Set(['positional', 'skeletal (chain branching)', 'skeletal (ring system)']);
const TYPE_NAME = { positional: 'positional', 'skeletal (chain branching)': 'chain-branching', 'skeletal (ring system)': 'ring-system', 'ring-chain': 'ring–chain',
  'functional group': 'functional-group', 'connectivity (metamer)': 'metameric', 'unsaturation type': 'unsaturation' };
const TYPE_ROW = { positional: 'Positional isomers', 'skeletal (chain branching)': 'Chain-branching isomers', 'skeletal (ring system)': 'Ring-system isomers', 'ring-chain': 'Ring–chain isomers',
  'functional group': 'Functional-group isomers', 'connectivity (metamer)': 'Metamers', 'unsaturation type': 'Unsaturation isomers' };
const BIN_EDGES = [1.05, 1.12, 1.25, 1.65];
const BIN_LABELS = ['1.00-1.05', '1.05-1.12', '1.12-1.25', '1.25-1.65', '>1.65'];
const EXAMPLES = [['o-xylene', 'p-xylene'], ['1-hexene', 'cyclohexane'], ['2,3-dimethylhexane', '3-ethylhexane']];
const EXTRA = { '2,3-dimethylhexane': 'CCCC(C)C(C)C', '3-ethylhexane': 'CCCC(CC)CC' };   // example names outside the dataset

const st = { RDKit: null, F: null, classify: null, W: null, meta: null, enet: null, solv: [], solutes: [], pairs: new Map(), val: null,
  byName: new Map(), byKey: new Map(), workers: [], ready: false, pending: false, res: null, shown: 50, open: null,
  filt: { IL: true, M: true, measured: false, hideOut: false, q: '' }, mol: { A: null, B: null } };

// ------------------------------------------------------------------ boot
async function getJSON(p) { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); }
function status(html) { $('#status').innerHTML = html; }
async function boot() {
  route(); window.addEventListener('hashchange', route);
  const [solv, solutes, pairs, val, meta, elements] = await Promise.all(['data/solvents.json', 'data/solutes.json', 'data/pairs.json', 'data/validation.json', 'model/meta.json', 'js/elements.json'].map(getJSON));
  const num = (o) => Object.fromEntries(Object.entries(o).map(([k, x]) => [k, typeof x === 'string' && x !== '' && !Number.isNaN(+x) ? +x : x]));
  for (const k of ['calibration', 'calibration_all', 'summary', 'types']) val[k] = val[k].map(num);
  Object.assign(st, { solv, solutes, val, meta });
  for (const p of pairs) st.pairs.set(p.key, p);
  for (const s of solutes) { st.byKey.set(s.inchikey, s); for (const n of [s.name, ...s.syn]) if (!st.byName.has(n.toLowerCase())) st.byName.set(n.toLowerCase(), s); }
  for (const [n, smi] of Object.entries(EXTRA)) if (!st.byName.has(n)) st.byName.set(n, { name: n, smiles: smi, syn: [], extra: true });
  $('#dl').innerHTML = solutes.flatMap((s) => [s.name, ...s.syn.slice(0, 2)]).filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a.localeCompare(b)).map((n) => `<option value="${esc(n)}">`).join('');
  st.RDKit = await window.initRDKitModule({ locateFile: () => 'vendor/RDKit_minimal.wasm' });
  st.F = makeFeaturizer(st.RDKit, elements); st.classify = makeIsoType(st.RDKit);
  renderMethod(); renderPair();
  for (const k of ['A', 'B']) { const el = $('#in' + k); el.addEventListener('change', () => showMol(k)); el.addEventListener('blur', () => showMol(k)); }
  $('#inT').addEventListener('input', () => { const T = +$('#inT').value; $('#tC').textContent = Number.isFinite(T) ? `${(T - 273.15).toFixed(1)} °C` : ''; });
  $('#inT').addEventListener('change', () => { if (st.res) rerunTemperature().catch(fail); });
  $('#go').addEventListener('click', () => screen().catch(fail));
  document.querySelectorAll('[data-ex]').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); const [x, y] = EXAMPLES[+a.dataset.ex]; $('#inA').value = x; $('#inB').value = y; showMol('A'); showMol('B'); screen().catch(fail); }));
  $('#go').disabled = false; status('');
  startEngine().catch((e) => { loading(null); status(`<span class="err-in">The prediction models could not be loaded (${esc(e.message)}). Reload the page to try again.</span>`); });
}

// ------------------------------------------------------------------ engine: download, workers
// loading indicator under the button: one sentence and a progress bar
function loading(text, frac) {
  const el = $('#load'); if (text == null) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden'); $('#loadText').textContent = st.pending ? 'The prediction models are still loading; the screen will start as soon as they are ready.' : text;
  if (frac != null) $('#loadBar').style.width = `${Math.max(2, Math.min(100, 100 * frac))}%`;
}
async function readAll(resp, onBytes) {
  const reader = resp.body.getReader(); const chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.length; onBytes(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out.buffer;
}
const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} did not respond within ${ms / 1000} s`)), ms))]);
async function startEngine() {
  const meta = st.meta, nk = meta.seeds.length, S = st.solv.length;
  const onnxNames = []; for (let k = 0; k < nk; k++) for (const p of ['dmpnn_prep', 'dmpnn_head', 'ghgnn_prep', 'ghgnn_head']) onnxNames.push(`${p}_${k}`);
  const files = [...onnxNames.map((n) => `model/onnx/${n}.onnx.bin`), 'model/enc.f32.bin', 'model/enet.f16.bin', 'data/emb_dmpnn.f16.bin', 'data/emb_ghgnn.f16.bin', 'vendor/ort-wasm-simd-threaded.wasm'];
  loading('Downloading the prediction models', 0);
  const resps = await Promise.all(files.map(async (f) => { const r = await fetch(f); if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r; }));
  const total = resps.reduce((a, r) => a + (+r.headers.get('content-length') || 0), 0) || 67e6; let got = 0;
  const tick = (b) => { got += b; loading(`Downloading the prediction models, ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`, 0.85 * got / total); };
  const bufs = await Promise.all(resps.map((r) => readAll(r, tick)));
  const B = Object.fromEntries(files.map((f, i) => [f, bufs[i]]));
  st.W = loadWeights({ index: meta.enc_index }, new Float32Array(B['model/enc.f32.bin']));
  const enet2 = f16ToF32(new Uint16Array(B['model/enet.f16.bin']));
  st.enet = meta.seeds.map((_, k) => { const [off, shape] = meta.enet_index[`g${k}/enet2/W`]; return { W2: enet2.subarray(off, off + shape[0] * shape[1]), w1: st.W[`g${k}/enet1/W`].d, b1: st.W[`g${k}/enet1/b`].d, b2: st.W[`g${k}/enet2/b`].d }; });
  const dm16 = new Uint16Array(B['data/emb_dmpnn.f16.bin']), gh16 = new Uint16Array(B['data/emb_ghgnn.f16.bin']);
  const sliceOf = (lo, hi) => { const n = hi - lo, dm = new Float32Array(nk * n * 256), gh = new Float32Array(nk * n * 226);
    for (let k = 0; k < nk; k++) { dm.set(f16ToF32(dm16.subarray((k * S + lo) * 256, (k * S + hi) * 256)), k * n * 256); gh.set(f16ToF32(gh16.subarray((k * S + lo) * 226, (k * S + hi) * 226)), k * n * 226); }
    return { lo: 0, hi: n, dm, gh, hba: st.solv.slice(lo, hi).map((s) => s.hba), hbd: st.solv.slice(lo, hi).map((s) => s.hbd) }; };
  const onnxCopy = () => Object.fromEntries(onnxNames.map((nm) => [nm, B[`model/onnx/${nm}.onnx.bin`].slice(0)]));
  const nW = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 1));   // each part holds its own copy of the model files
  let ready = 0; loading(`Starting the calculation engine, 0 of ${nW} parts`, 0.86);
  try {
    const per = Math.ceil(S / nW); const jobs = [];
    for (let w = 0; w < nW; w++) {
      const lo = w * per, hi = Math.min(S, lo + per); if (hi <= lo) break;
      const wk = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }); wk.lo = lo; wk.hi = hi; st.workers.push(wk);
      const fail = new Promise((_, rej) => wk.addEventListener('error', (e) => rej(new Error(e.message || 'the worker could not start'))));
      const slice = sliceOf(lo, hi); const onnx = onnxCopy(); const ortWasm = B['vendor/ort-wasm-simd-threaded.wasm'].slice(0);
      jobs.push(withTimeout(Promise.race([call(wk, { type: 'init', onnx, meta, slice, ortWasm }, [slice.dm.buffer, slice.gh.buffer, ortWasm, ...Object.values(onnx)]), fail]), 120000, 'A worker')
        .then(() => { ready++; loading(`Starting the calculation engine, ${ready} of ${nW} parts`, 0.86 + 0.14 * ready / nW); }));
    }
    await Promise.all(jobs);
  } catch (e) {
    // Fallback: run the same engine on the page itself (slower, but works where module workers are blocked)
    console.warn('Workers unavailable, running on the page:', e);
    st.workers.forEach((w) => w.terminate()); st.workers = [];
    loading('Starting the calculation engine on this page', 0.9);
    const ort = await import('../vendor/ort.wasm.bundle.min.js'); const { SliceScreener } = await import('./screen.js');
    ort.env.wasm.numThreads = 1; ort.env.wasm.wasmBinary = new Uint8Array(B['vendor/ort-wasm-simd-threaded.wasm']);
    ort.env.wasm.wasmPaths = { mjs: new URL('../vendor/ort-wasm-simd-threaded.js', import.meta.url).href };
    const scr = await SliceScreener.create(ort, Object.fromEntries(Object.entries(onnxCopy()).map(([k, v]) => [k, new Uint8Array(v)])), meta, sliceOf(0, S), null);
    st.workers = [{ lo: 0, hi: S, local: scr }];
  }
  st.ready = true; loading(null);
  status('Ready. Everything runs in your browser; structures you enter are not sent anywhere.');
  if (st.pending) { st.pending = false; screen().catch(fail); }
}
let callId = 0;
function call(wk, msg, transfer = []) {
  if (wk.local) return (async () => { wk.local.setEdgeWeights(msg.edge || []); return { res: await wk.local.run(msg.solutes, msg.T, msg.ghgnn) }; })();
  return new Promise((resolve, reject) => {
    const id = ++callId; msg.id = id;
    const h = (e) => { if (e.data.id !== id) return; wk.removeEventListener('message', h); e.data.type === 'error' ? reject(new Error(e.data.message)) : resolve(e.data); };
    wk.addEventListener('message', h); wk.postMessage(msg, transfer);
  });
}
function edgeWeight(k, v) {
  const { w1, b1, W2, b2 } = st.enet[k]; const D2 = b2.length; const out = Float32Array.from(b2);
  for (let j = 0; j < 32; j++) { const h = Math.max(0, v * w1[j] + b1[j]); if (!h) continue; const o = j * D2; for (let i = 0; i < D2; i++) out[i] += h * W2[o + i]; }
  return out;
}

function fail(e) { console.error(e); loading(null); $('#go').disabled = false; st.pending = false;
  $('#error').textContent = `The screen stopped: ${e && e.message ? e.message : e}. Check the two structures and try again.`; status(''); }

// ------------------------------------------------------------------ molecules
function resolveInput(text) {
  const t = (text || '').trim(); if (!t) return { error: 'Enter a name or a SMILES string.' };
  const hit = st.byName.get(t.toLowerCase());
  const smiles = hit ? hit.smiles : t;
  const P = st.F.parse(smiles);
  if (!P) return { error: hit ? `Could not read the structure of ${t}.` : `“${t}” is neither a known isomer name nor a valid SMILES string.` };
  if (P.smiles.includes('.')) return { error: 'Enter a single molecule (no salts or mixtures).' };
  if (P.atoms.some((a) => a.chg)) return { error: 'Charged molecules are not supported; enter the neutral isomer.' };
  const ik = P.inchikey; let ds = st.byKey.get(ik); let note = '';
  if (!ds && ik) { const same = st.solutes.filter((s) => s.inchikey.slice(0, 14) === ik.slice(0, 14)); if (same.length === 1) { ds = same[0]; note = `matched to ${ds.name} in the dataset (stereochemistry ignored)`; } }
  return { P, smiles: ds ? ds.smiles : P.smiles, name: ds ? prettyName(ds, t) : (hit ? hit.name : P.canonical), ds, note, ik: ds ? ds.inchikey : ik, formula: formula(P) };
}
function formula(P) {
  const sym = (z) => ({ 1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I' }[z] || `Z${z}`);
  const c = {}; for (const a of P.atoms) { c[sym(a.z)] = (c[sym(a.z)] || 0) + 1; if (a.hs) c.H = (c.H || 0) + a.hs; }
  const keys = Object.keys(c).sort((a, b) => (a === 'C' ? -1 : b === 'C' ? 1 : a === 'H' ? -1 : b === 'H' ? 1 : a.localeCompare(b)));
  return keys.map((k) => k + (c[k] > 1 ? c[k] : '')).join('');
}
// show the name the user typed when it is a known synonym (o-xylene rather than 1,2-dimethylbenzene)
function prettyName(ds, typed) { const t = typed.trim(); return [ds.name, ...ds.syn].some((n) => n.toLowerCase() === t.toLowerCase()) ? t : ds.name; }
const fmlHTML = (f) => f.replace(/(\d+)/g, '<sub>$1</sub>');
function svgOf(smiles, w = 120, h = 96) {
  try { const m = st.RDKit.get_mol(smiles); const s = m.get_svg_with_highlights(JSON.stringify({ width: w, height: h, bondLineWidth: 1.2, backgroundColour: [1, 1, 1, 0], clearBackground: false, fixedBondLength: 26, padding: 0.08 })); m.delete(); return s; }
  catch (e) { try { const m = st.RDKit.get_mol(smiles); const s = m.get_svg(w, h); m.delete(); return s; } catch (e2) { return ''; } }
}
function showMol(k) {
  const d = $('#draw' + k), m = $('#meta' + k); const v = $('#in' + k).value;
  if (!v.trim() || !st.F) { d.innerHTML = '<span class="ph">Structure appears here</span>'; m.innerHTML = ''; st.mol[k] = null; return; }
  const r = resolveInput(v); st.mol[k] = r;
  if (r.error) { d.innerHTML = ''; m.innerHTML = `<span class="err-in">${esc(r.error)}</span>`; return; }
  d.innerHTML = svgOf(r.smiles, 220, 160);
  m.innerHTML = `<b>${fmlHTML(r.formula)}</b>${r.ds ? ', measured in the dataset' : ', not in the dataset'}${r.note ? `<br>${esc(r.note)}` : ''}`;
}
function encode(r) {
  const nk = st.meta.seeds.length; const dg = st.F.dmpnnGraph(r.P); let gg = null, why = '';
  try { gg = st.F.ghgnnGraph(r.P); } catch (e) { why = e.message; }
  return { dm: Array.from({ length: nk }, (_, k) => Float32Array.from(dmpnnEncode(st.W, k, dg))),
           gh: gg ? Array.from({ length: nk }, (_, k) => Float32Array.from(ghgnnEncode(st.W, k, gg, st.meta.ghgnn.eps))) : null, hba: gg ? gg.hba : 0, hbd: gg ? gg.hbd : 0, why };
}

// ------------------------------------------------------------------ screening
async function screen() {
  $('#error').textContent = ''; showMol('A'); showMol('B');
  const A = st.mol.A, B = st.mol.B;
  if (!A || !B) return void ($('#error').textContent = 'Enter both isomers.');
  if (A.error || B.error) return void ($('#error').textContent = A.error || B.error);
  if (A.formula !== B.formula) return void ($('#error').textContent = `These are not constitutional isomers: ${A.formula} and ${B.formula} have different molecular formulas.`);
  if (A.ik && B.ik && A.ik.slice(0, 14) === B.ik.slice(0, 14)) return void ($('#error').textContent = 'These two structures have the same connectivity (identical compounds or stereoisomers). IsoScreen handles constitutional isomers only.');
  if (!st.ready) { st.pending = true; loading('', null); return; }
  const T = clampT(); $('#go').disabled = true; status('Screening the solvent library'); loading('Screening 6,911 solvents', 0.05); const t0 = performance.now();
  const encA = encode(A), encB = encode(B); const ghOK = !!(encA.gh && encB.gh);
  const key = ['MOL_' + A.ik, 'MOL_' + B.ik].sort().join('|'); const pair = st.pairs.get(key) || null;
  const out = await runWorkers([encA, encB], [T], ghOK);
  st.res = { A, B, encA, encB, ghOK, pair, T, out, rel: pair ? pair.rel : st.classify(A.smiles, B.smiles), t: performance.now() - t0 };
  st.shown = 50; st.open = null; $('#go').disabled = false; loading(null);
  status(`Screened ${st.solv.length.toLocaleString()} solvents in ${(st.res.t / 1000).toFixed(1)} s, in your browser.`);
  renderResults(); renderPair();
}
async function rerunTemperature() {
  const r = st.res; const T = clampT(); if (!st.ready || T === r.T) return; status('Recalculating for the new temperature'); loading('Recalculating for the new temperature', 0.1);
  const dm = await runWorkers([r.encA, r.encB], [T], false);
  for (let u = 0; u < 2; u++) r.out[u].dm = dm[u].dm; r.T = T; loading(null); renderResults(); renderPair(); status(`Recalculated at ${tK(T)} K.`);
}
function clampT() { let T = +$('#inT').value; if (!Number.isFinite(T)) T = 323.15; T = Math.min(423.15, Math.max(273.15, T)); $('#inT').value = T; $('#tC').textContent = `${(T - 273.15).toFixed(1)} °C`; return T; }
async function runWorkers(sol, T, gh) {
  const nk = st.meta.seeds.length; const cache = new Map(); const W = (k, v) => { const key = k + ':' + v; if (!cache.has(key)) cache.set(key, edgeWeight(k, v)); return cache.get(key); };
  const edgeFor = (wk) => { if (!gh) return []; const vs = new Set(); for (const u of sol) for (let i = wk.lo; i < wk.hi; i++) { const s = st.solv[i]; vs.add(Math.min(s.hba, u.hbd) + Math.min(u.hba, s.hbd)); }
    const out = []; for (let k = 0; k < nk; k++) for (const v of vs) out.push([k, v, W(k, v)]); return out; };
  let done = 0; const nW = st.workers.length;
  const parts = await Promise.all(st.workers.map((wk) => call(wk, { type: 'run', solutes: sol.map((s) => ({ dm: s.dm, gh: gh ? s.gh : null, hba: s.hba, hbd: s.hbd })), T, ghgnn: gh, edge: edgeFor(wk) })
    .then((r) => { done++; loading(`Screening ${st.solv.length.toLocaleString()} solvents`, 0.1 + 0.9 * done / nW); return r; })));
  const S = st.solv.length, K = T.length;
  return sol.map((_, u) => {
    const dm = new Float32Array(nk * S * K), A = new Float32Array(nk * S).fill(NaN), B = new Float32Array(nk * S).fill(NaN);
    parts.forEach((p, w) => { const wk = st.workers[w], n = wk.hi - wk.lo, r = p.res[u];
      for (let k = 0; k < nk; k++) { dm.set(r.dm.subarray(k * n * K, (k + 1) * n * K), (k * S + wk.lo) * K); A.set(r.A.subarray(k * n, (k + 1) * n), k * S + wk.lo); B.set(r.B.subarray(k * n, (k + 1) * n), k * S + wk.lo); } });
    return { dm, A, B };
  });
}

// ------------------------------------------------------------------ results table
function calib(regime, S) { const i = BIN_EDGES.findIndex((e) => S < e); const lab = BIN_LABELS[i < 0 ? 4 : i]; return st.val.calibration.find((c) => c.regime === regime && c.predicted_S_bin === lab); }
function summary(regime) { return st.val.summary.find((s) => s.regime === regime && s.model === 'AVG'); }
function domRes(dom, kind) { const d = st.val.domain.find((x) => x.domain === dom && x.kind === (kind === 'IL' ? 'ionic liquids' : 'molecular solvents')); return d ? d.resolution : null; }
function measuredFor(pair, si, T, flip) {
  if (!pair) return null; const m = pair.m.filter((x) => x[0] === si); if (!m.length) return null;
  const best = Math.min(...m.map((x) => Math.abs(x[1] - T))); const sel = m.filter((x) => Math.abs(Math.abs(x[1] - T) - best) < 0.5);
  const d = sel.reduce((a, x) => a + x[2], 0) / sel.length * (flip ? -1 : 1);
  return { d, S: Math.exp(Math.abs(d)), T: sel[0][1], src: sel.map((x) => ({ lab: x[3], url: x[4], note: x[5] })), n: m.length };
}
function rows() {
  const r = st.res, S = st.solv.length, nk = st.meta.seeds.length, T = r.T; const flip = r.pair ? ('MOL_' + r.A.ik) !== r.pair.a : false;
  const mean = (arr, i, f) => { let s = 0; for (let k = 0; k < nk; k++) s += f(arr, k * S + i); return s / nk; };
  const out = [];
  for (let i = 0; i < S; i++) {
    const dA = mean(r.out[0].dm, i, (a, j) => a[j]), dB = mean(r.out[1].dm, i, (a, j) => a[j]);
    const gA = r.ghOK ? mean(r.out[0].A, i, (a, j) => a[j] + r.out[0].B[j] / T) : NaN, gB = r.ghOK ? mean(r.out[1].A, i, (a, j) => a[j] + r.out[1].B[j] / T) : NaN;
    const lA = r.ghOK ? (dA + gA) / 2 : dA, lB = r.ghOK ? (dB + gB) / 2 : dB; const d = lA - lB, Sv = Math.exp(Math.abs(d));
    const s = st.solv[i]; const known = !!r.pair && s.domain !== 'outside domain'; const regime = known ? 'known_isomers_new_solvents' : 'new_isomers';
    const c = calib(regime, Sv); let label, tier;
    if (known) { label = `${pct(c.hit_rate)} confirmed`; tier = c.hit_rate >= 0.85 ? 2 : c.hit_rate >= 0.6 ? 1 : 0; }
    else if (Sv < 1.12) { label = 'Unresolved'; tier = 0; }
    else if (Sv < 1.65) { label = `Indicative (${pct(c.hit_rate)} confirmed)`; tier = 1; }
    else { label = `Likely real (${pct(c.hit_rate)} confirmed)`; tier = 2; }
    const meas = measuredFor(r.pair, i, T, flip);
    out.push({ i, s, lA, lB, d, S: Sv, fav: d > 0 ? 'B' : 'A', cap: Math.exp(-Math.min(lA, lB)), label, tier, known, meas,
      S_dm: Math.exp(Math.abs(dA - dB)), fav_dm: dA - dB > 0 ? 'B' : 'A', S_gh: Math.exp(Math.abs(gA - gB)), fav_gh: gA - gB > 0 ? 'B' : 'A', lnA: [dA, gA], lnB: [dB, gB] });
  }
  out.sort((a, b) => b.S - a.S); out.forEach((x, k) => (x.rankAll = k + 1));
  return out;
}
function filtered(all) {
  const f = st.filt, q = f.q.trim().toLowerCase();
  return all.filter((x) => (x.s.kind === 'IL' ? f.IL : f.M) && (!f.measured || x.s.measured) && (!f.hideOut || x.s.domain !== 'outside domain')
    && (!q || x.s.name.toLowerCase().includes(q) || (x.s.short || '').toLowerCase().includes(q)));
}
const sname = (s) => s.short || s.name;
const typeLabel = (s) => (s.kind === 'IL' ? `${s.cls || 'other'} ionic liquid` : 'molecular solvent');
const tK = (T) => (Math.abs(T - Math.round(T)) < 1e-9 ? String(T) : T.toFixed(2));
function renderResults() {
  const r = st.res, A = r.A, B = r.B; const all = rows(); const list = filtered(all); r.rows = all;
  const known = !!r.pair; const sk = summary('known_isomers_new_solvents'), sn = summary('new_isomers');
  const nA = esc(A.name), nB = esc(B.name);
  const nsep = list.filter((x) => x.S > 1.12).length;
  let html = `<h2 class="headline">${nsep.toLocaleString()} of ${list.length.toLocaleString()} solvents are predicted to separate ${nA} and ${nB} at ${tK(r.T)} K, with a selectivity above 1.12</h2>`;
  if (known) html += `<p class="context known">This pair has been measured in ${r.pair.n_solv} solvents. For the other solvents, predicted selectivities are typically within ${Math.round((sk.resolution - 1) * 100)}% of measured values and can be used directly; where a measurement exists, it takes precedence. <a href="#pair">All ${r.pair.n_meas.toLocaleString()} measurements</a></p>`;
  else html += `<p class="context new">This pair has not been measured. Predicted selectivities below 1.12 cannot be told apart from no difference; read larger values as a level of confidence, not as a value.</p>`;
  const notes = [];
  if (!known && LEAST.has(r.rel)) notes.push(`These are ${TYPE_NAME[r.rel]} isomers, the type for which predictions are least reliable.`);
  if (!known) notes.push('Measuring this pair in a few of the top-ranked, structurally different solvents should turn it into a measured pair, for which predictions are far more precise. This follows from the validation but has not been tested directly.');
  const ss = known ? sk : sn; const ratio = ss.median_tests_to_best / ss.median_tests_random;
  notes.push(`Test the solvents in the order shown: in validation this order found the best solvent after ${ratio < 0.36 ? 'about a third' : 'fewer than half'} of the tests needed in random order.`);
  if (!r.ghOK) notes.push(`GH-GNN cannot represent this molecule (${esc(r.encA.why || r.encB.why)}), so values come from D-MPNN alone and are less reliable.`);
  if (known && (r.T < r.pair.Tmin - 0.5 || r.T > r.pair.Tmax + 0.5)) notes.push(`${tK(r.T)} K lies outside the temperatures at which this pair was measured (${r.pair.Tmin}–${r.pair.Tmax} K).`);
  html += notes.map((n) => `<p class="aside">${n}</p>`).join('');
  html += `<figure>${chart(list.slice(0, 30))}</figure>`;
  html += `<div class="toolbar">
    <label><input type="checkbox" data-f="IL" ${st.filt.IL ? 'checked' : ''}>Ionic liquids</label>
    <label><input type="checkbox" data-f="M" ${st.filt.M ? 'checked' : ''}>Molecular solvents</label>
    <label><input type="checkbox" data-f="measured" ${st.filt.measured ? 'checked' : ''}>Measured solvents only</label>
    <label><input type="checkbox" data-f="hideOut" ${st.filt.hideOut ? 'checked' : ''}>Hide solvents outside the models' range</label>
    <input type="search" id="fq" placeholder="Find a solvent" value="${esc(st.filt.q)}" aria-label="Find a solvent">
    <span class="sp"></span><button class="quiet" id="csv">Download table</button><button class="quiet" id="copy">Copy table</button></div>`;
  html += `<div class="tbl"><table><thead><tr><th class="num">Rank</th><th>Solvent</th><th class="num">Predicted S<sup>∞</sup></th><th>Dissolves more</th><th>Confidence</th><th class="num">Measured S<sup>∞</sup></th><th class="num">Capacity 1/γ<sup>∞</sup></th><th></th></tr></thead><tbody>`;
  const favName = (f) => esc(f === 'A' ? A.name : B.name);
  list.slice(0, st.shown).forEach((x, n) => {
    const fl = x.s.domain === 'near domain' ? '<span class="flag">near the models’ range</span>' : x.s.domain === 'outside domain' ? '<span class="flag out">outside the models’ range</span>' : '';
    const opp = x.meas && x.meas.S > 1.026 && (x.meas.d > 0) !== (x.d > 0);
    const flagged = x.meas && x.meas.src.every((m) => m.note);
    const ms = x.meas ? `${fmt(x.meas.S)}${flagged ? '<div class="sub">inconsistent value, see details</div>' : ''}${opp ? `<div class="sub">favours ${favName(x.meas.d > 0 ? 'B' : 'A')}</div>` : ''}${Math.abs(x.meas.T - r.T) > 0.5 ? `<div class="sub">at ${x.meas.T} K</div>` : ''}` : '';
    html += `<tr class="r${st.open === x.i ? ' open' : ''}" data-i="${x.i}" tabindex="0"><td class="num rk">${n + 1}</td><td>${esc(sname(x.s))}<div class="sub">${esc(typeLabel(x.s))}</div></td><td class="num sv">${fmt(x.S)}</td><td>${favName(x.fav)}</td>
      <td><span class="lv lv${x.tier}">${x.label}</span></td><td class="num">${ms}</td><td class="num">${fmtCap(x.cap)}</td><td>${fl}</td></tr>`;
    if (st.open === x.i) html += `<tr class="d"><td></td><td colspan="7">${details(x)}</td></tr>`;
  });
  html += '</tbody></table></div>';
  if (list.length > st.shown) html += `<div class="more"><button class="quiet" id="more">Show 50 more of ${(list.length - st.shown).toLocaleString()}</button></div>`;
  html += `<p class="foot">Selectivity S<sup>∞</sup> is the ratio of the infinite-dilution activity coefficients of the two isomers, larger over smaller. The solvent dissolves more of the isomer with the smaller activity coefficient. Predictions are the mean of two models, each averaged over three training runs. Confirmed is the share of validation cases in which the measured selectivity was at least 1.05 in the predicted direction. Capacity is 1/γ<sup>∞</sup> of the isomer the solvent dissolves more; single activity coefficients are typically within ${pct(st.val.typical_gamma_error.new_isomers)} to ${pct(st.val.typical_gamma_error.known_isomers_new_solvents)} of measured values. <a href="#method">How these numbers were obtained</a></p>`;
  $('#results').innerHTML = html;
  document.querySelectorAll('[data-f]').forEach((c) => c.addEventListener('change', () => { st.filt[c.dataset.f] = c.checked; st.shown = 50; renderResults(); }));
  const fq = $('#fq'); fq.addEventListener('input', () => { st.filt.q = fq.value; st.shown = 50; const pos = fq.selectionStart; renderResults(); const f2 = $('#fq'); f2.focus(); f2.setSelectionRange(pos, pos); });
  $('#more')?.addEventListener('click', () => { st.shown += 50; renderResults(); });
  $('#csv').addEventListener('click', () => { const blob = new Blob([csv(list)], { type: 'text/csv' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `isoscreen_${slug(A.name)}_${slug(B.name)}_${tK(r.T)}K.csv`; document.body.appendChild(a); a.click(); a.remove(); });
  $('#copy').addEventListener('click', async () => { try { await navigator.clipboard.writeText(csv(list)); $('#copy').textContent = 'Copied'; } catch (e) { $('#copy').textContent = 'Copying is blocked here'; } });
  const toggle = (tr) => { const i = +tr.dataset.i; st.open = st.open === i ? null : i; renderResults(); };
  document.querySelectorAll('tr.r').forEach((tr) => { tr.addEventListener('click', () => toggle(tr)); tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') toggle(tr); }); });
}
const slug = (s) => String(s).replace(/[^A-Za-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40);
function details(x) {
  const r = st.res, s = x.s; const fav = (f) => esc(f === 'A' ? r.A.name : r.B.name);
  const src = { dataset: 'Measured solvent in the dataset', 'ion combination': 'Ionic liquid combined from cations and anions in the dataset; not yet measured as a solvent', CHEM21: 'CHEM21 solvent selection guide (Prat et al., Green Chem. 2016)', 'Langner 2022': 'List of common organic solvents (Langner and Brabec, Mendeley Data 2022)' }[s.src] || s.src;
  const kv = [['Full name', esc(s.name)], ['SMILES', esc(s.smiles)], s.cas ? ['CAS', esc(s.cas)] : null, ['Source', esc(src)],
    ['Similarity to measured solvents', `${fmt(s.sim, 2)}${s.domain === 'near domain' ? '. Selectivity errors can be about twice as large for such solvents.' : s.domain === 'outside domain' ? '. Unlike the measured solvents; read with the rules for new pairs.' : ''}`],
    ['D-MPNN alone', `S<sup>∞</sup> ${fmt(x.S_dm)}, dissolves more ${fav(x.fav_dm)}`], r.ghOK ? ['GH-GNN alone', `S<sup>∞</sup> ${fmt(x.S_gh)}, dissolves more ${fav(x.fav_gh)}`] : null,
    ['ln γ<sup>∞</sup>', `${esc(r.A.name)} ${fmt(x.lA)}, ${esc(r.B.name)} ${fmt(x.lB)}`],
    x.meas ? ['Measured', `S<sup>∞</sup> ${fmt(x.meas.S)} at ${x.meas.T} K, dissolves more ${fav(x.meas.d > 0 ? 'B' : 'A')}. ${x.meas.src.map((m) => (m.url ? `<a href="${esc(m.url)}" target="_blank" rel="noopener">${esc(m.lab)}</a>` : esc(m.lab)) + (m.note ? ' (inconsistent with the other data, see the paper, SI S1.4)' : '')).join('; ')}`] : null].filter(Boolean);
  return `<div class="dg"><div>${svgOf(s.smiles, 180, 130)}</div><div class="kv">${kv.map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join('')}</div></div>`;
}
function csv(list) {
  const r = st.res; const q = (v) => { let t = String(v ?? ''); if (/^[=+\-@]/.test(t) && !/^-?\d/.test(t)) t = "'" + t; return `"${t.replace(/"/g, '""')}"`; };
  const head = ['rank', 'solvent', 'abbreviation', 'type', 'smiles', 'cas', 'predicted_S_inf', 'dissolves_more', 'confidence', 'measured_S_inf', 'measured_T_K', 'measured_dissolves_more', 'capacity_1_over_gamma',
    `ln_gamma_${r.A.name}`, `ln_gamma_${r.B.name}`, 'S_inf_DMPNN', 'S_inf_GHGNN', 'models_range', 'similarity_to_measured_solvents', 'solvent_source'];
  const L = [head.map(q).join(',')];
  list.forEach((x, n) => L.push([n + 1, x.s.name, x.s.short, typeLabel(x.s), x.s.smiles, x.s.cas, x.S.toFixed(4), x.fav === 'A' ? r.A.name : r.B.name, x.label,
    x.meas ? x.meas.S.toFixed(4) : '', x.meas ? x.meas.T : '', x.meas ? (x.meas.d > 0 ? r.B.name : r.A.name) : '', x.cap.toFixed(4), x.lA.toFixed(4), x.lB.toFixed(4),
    x.S_dm.toFixed(4), r.ghOK ? x.S_gh.toFixed(4) : '', x.s.domain, x.s.sim, x.s.src].map(q).join(',')));
  return `# IsoScreen ${new Date().toISOString().slice(0, 10)}; ${r.A.name} / ${r.B.name}; ${tK(r.T)} K; ${r.pair ? 'measured pair' : 'pair not measured'}\n` + L.join('\n');
}
function chart(top) {
  if (!top.length) return '<p class="aside">No solvents match the filters.</p>';
  const W = 980, rowH = 18, L = 250, R = 20, T0 = 26, plotH = top.length * rowH, H = T0 + plotH + 44;
  const orient = (x) => Math.exp(x.meas.d * Math.sign(x.d));
  const vals = top.flatMap((x) => (x.meas ? [x.S, orient(x)] : [x.S]));
  const lo = Math.min(1, ...vals) / 1.01, hi = Math.max(1.8, ...vals) * 1.04;
  const X = (v) => L + (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (W - L - R);
  const yb = T0 + plotH + 6; let g = '';
  // reference lines, labelled at the top unless they would collide
  let lastX = -1e9;
  for (const v of [1.026, 1.12, 1.65]) { if (v <= lo || v >= hi) continue; const x = X(v);
    g += `<line x1="${x}" x2="${x}" y1="${T0 - 4}" y2="${yb}" stroke="#D3D1C7"/>`;
    if (x - lastX > 34) { g += `<text class="g" x="${x}" y="${T0 - 9}" text-anchor="middle">${v}</text>`; lastX = x; } }
  // axis
  const ticks = [1, 1.1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100].filter((v) => v >= lo && v <= hi);
  g += `<line x1="${L}" x2="${W - R}" y1="${yb}" y2="${yb}" stroke="#2C2C2A"/>` + ticks.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${yb}" y2="${yb + 4}" stroke="#2C2C2A"/><text x="${X(v)}" y="${yb + 17}" text-anchor="middle">${v}</text>`).join('');
  g += `<text class="g" x="${L}" y="${yb + 36}">Selectivity S∞, log scale. Lines at 1.026 (experimental resolution), 1.12 and 1.65.</text>`;
  g += `<line x1="${L}" x2="${L}" y1="${T0 - 4}" y2="${yb}" stroke="#2C2C2A"/>`;
  let labP = false, labM = false;
  top.forEach((x, k) => {
    const y = T0 + k * rowH + rowH / 2; const nm = sname(x.s); const lab = nm.length > 34 ? nm.slice(0, 33) + '…' : nm;
    g += `<text x="${L - 10}" y="${y + 4}" text-anchor="end">${esc(lab)}</text>`;
    const xp = X(x.S); g += `<circle cx="${xp}" cy="${y}" r="3.6" fill="#2C2C2A"/>`;
    if (x.meas) { const xm = X(orient(x)); g += `<circle cx="${xm}" cy="${y}" r="4.6" fill="none" stroke="#534AB7" stroke-width="1.4"/>`;
      if (!labM && k < top.length - 1) { g += `<text class="p" x="${Math.max(xm, xp) + 10}" y="${y + 4}">measured</text>`; labM = true; } }
    if (!labP && !x.meas) { g += `<text x="${xp + 9}" y="${y + 4}" class="g">predicted</text>`; labP = true; }
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Predicted selectivity of the top ${top.length} solvents">${g}</svg>`;
}

// ------------------------------------------------------------------ measured-data page
function renderPair() {
  const el = $('#v-pair'); const r = st.res;
  if (!r) { el.innerHTML = '<h1 class="page">Measured data</h1><p class="prose">Screen an isomer pair first. If the pair is in the dataset, every measured selectivity is listed here with its source.</p>'; return; }
  if (!r.pair) { el.innerHTML = `<h1 class="page">${esc(r.A.name)} and ${esc(r.B.name)} have not been measured together</h1><p class="prose">The dataset holds 31,541 selectivities of 359 isomer pairs, each from two isomers measured in the same study, solvent and temperature. This pair is not among them, so the screen uses the rules for new pairs.</p>`; return; }
  const p = r.pair, flip = ('MOL_' + r.A.ik) !== p.a; const m = p.m.map((x) => ({ s: typeof x[0] === 'number' ? st.solv[x[0]] : { name: x[0], short: '' }, T: x[1], d: x[2] * (flip ? -1 : 1), lab: x[3], url: x[4], note: x[5] }));
  m.forEach((x) => (x.S = Math.exp(Math.abs(x.d)))); m.sort((a, b) => b.S - a.S);
  const best = m.find((x) => !x.note) || m[0]; const anyNote = m.some((x) => x.note);
  let h = `<h1 class="page">${esc(r.A.name)} and ${esc(r.B.name)} reach a selectivity of ${fmt(best.S, 2)} in ${esc(best.s.short || best.s.name)}</h1>
    <p class="prose">${p.n_meas.toLocaleString()} measurements in ${p.n_solv} solvents between ${p.Tmin} and ${p.Tmax} K. These are ${esc(TYPE_NAME[p.rel] || p.rel)} isomers.</p>`;
  if (anyNote) h += '<p class="aside">† Cyclohexane values from this study are inconsistent with all other data in the set (see the paper, Supporting Information S1.4). They are listed unchanged but were not used to evaluate solvent ranking.</p>';
  h += `<div class="tbl" style="margin-top:22px"><table><thead><tr><th>Solvent</th><th class="num">T, K</th><th class="num">S<sup>∞</sup></th><th>Dissolves more</th><th>Source</th></tr></thead><tbody>`;
  for (const x of m) h += `<tr><td>${esc(x.s.short || x.s.name)}</td><td class="num">${x.T}</td><td class="num" style="white-space:nowrap">${x.note ? '† ' : ''}${fmt(x.S)}</td><td>${esc(x.d > 0 ? r.B.name : r.A.name)}</td><td class="sub">${x.url ? `<a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.lab)}</a>` : esc(x.lab)}</td></tr>`;
  el.innerHTML = h + '</tbody></table></div>';
}

// ------------------------------------------------------------------ method page
function renderMethod() {
  const v = st.val; const get = (reg, lab) => v.calibration.find((c) => c.regime === reg && c.predicted_S_bin === lab);
  const sk = summary('known_isomers_new_solvents'), sn = summary('new_isomers');
  const calRow = (reg, name) => `<tr><td>${name}</td>${BIN_LABELS.map((b) => `<td class="num">${pct(get(reg, b).hit_rate)}</td>`).join('')}</tr>`;
  const types = [...new Set(v.types.map((t) => t.isomer_type))].filter((t) => t !== 'unsaturation type');
  const tp = (reg, t) => v.types.find((x) => x.regime === reg && x.model === 'AVG' && x.isomer_type === t);
  const sumRow = (s, name) => `<tr><td>${name}</td><td class="num">${s.resolution.toFixed(3)}</td><td class="num">${pct(s.spurious_rate)}</td><td class="num">${pct(s.hit_rate)}</td><td class="num">${s.median_tests_to_best} of ${s.median_tests_random.toFixed(0)}</td></tr>`;
  const nMeas = st.solv.filter((s) => s.measured).length;
  $('#v-method').innerHTML = `<h1 class="page">Predictions are precise for measured pairs and a level of confidence for new ones</h1>
  <p>IsoScreen ranks solvents for separating two constitutional isomers by their selectivity at infinite dilution, S<sup>∞</sup>. How far a prediction can be trusted depends on whether the pair has been measured before. Four rules follow from the validation in the accompanying paper.</p>
  <ol><li><b>Measured pairs.</b> For a pair measured in other solvents, predicted selectivities in new solvents can be used as values. They are typically within ${Math.round((sk.resolution - 1) * 100)}% of measured values.</li>
  <li><b>New pairs.</b> For a pair never measured, a predicted selectivity below 1.12 cannot be told apart from no difference, and only predictions above about 1.65 should be trusted. Positional, chain-branching and ring-system isomers need extra caution.</li>
  <li><b>Order of tests.</b> In both cases the ranking is the most useful output. Testing solvents in the predicted order found the best one after a median of ${sk.median_tests_to_best} tests for measured pairs and ${sn.median_tests_to_best} for new pairs, against ${sk.median_tests_random.toFixed(0)} in random order.</li>
  <li><b>Measure a few, then screen.</b> Because the models resolve isomers they have seen, measuring a new pair in a few solvents may turn it into a measured pair. This follows from the results but has not been tested directly.</li></ol>
  <h2>How often a predicted separation is confirmed</h2>
  <p class="aside">Share of validation cases in which the measured selectivity was at least 1.05 in the predicted direction, by predicted selectivity. At most ${pct(maxHit())} is possible, because only that share of measured selectivities reaches 1.05.</p>
  <table><thead><tr><th>Predicted S<sup>∞</sup></th>${BIN_LABELS.map((b) => `<th class="num">${b.replace('-', '–')}</th>`).join('')}</tr></thead><tbody>${calRow('known_isomers_new_solvents', 'Measured pair, new solvent')}${calRow('new_isomers', 'New pair')}</tbody></table>
  <h2>By type of isomer</h2>
  <p class="aside">Confirmed share among predictions of at least 1.12.</p>
  <table><thead><tr><th>Isomer type</th><th class="num">Measured pair, new solvent</th><th class="num">New pair</th></tr></thead><tbody>${types.map((t) => `<tr><td>${esc(TYPE_ROW[t] || t)}</td><td class="num">${pct(tp('known_isomers_new_solvents', t).hit_rate)}</td><td class="num">${pct(tp('new_isomers', t).hit_rate)}</td></tr>`).join('')}</tbody></table>
  <h2>Overall</h2>
  <table><thead><tr><th></th><th class="num">Typical selectivity error</th><th class="num">False separations</th><th class="num">Confirmed</th><th class="num">Tests to find the best solvent</th></tr></thead><tbody>${sumRow(sk, 'Measured pair, new solvent')}${sumRow(sn, 'New pair')}</tbody></table>
  <p class="aside">Typical selectivity error is the median ratio of predicted to measured selectivity; independent experiments agree to 1.026. False separations: pairs that experiments cannot tell apart (below 1.026) but for which more than 1.12 was predicted. Tests: median over the 48 pairs measured in at least 30 solvents.</p>
  <h2>Solvents unlike the measured ones</h2>
  <p>Each solvent is compared with the ${nMeas} measured solvents by its highest structural similarity (Tanimoto similarity of Morgan fingerprints, radius 2). Selectivities were less precise for less similar solvents, and solvents below 0.3 are read with the rules for new pairs.</p>
  <table><thead><tr><th>Similarity</th><th class="num">Typical selectivity error, molecular solvents</th><th class="num">Ionic liquids</th></tr></thead><tbody>${['in domain', 'near domain', 'outside domain'].map((d) => { const a = v.domain.find((x) => x.domain === d && x.kind === 'molecular solvents'), b = v.domain.find((x) => x.domain === d && x.kind === 'ionic liquids'); return `<tr><td>${{ 'in domain': '0.5 and above', 'near domain': '0.3 to 0.5', 'outside domain': 'below 0.3' }[d]}</td><td class="num">${a.resolution?.toFixed(3) ?? '–'}</td><td class="num">${b.resolution?.toFixed(3) ?? '–'}</td></tr>`; }).join('')}</tbody></table>
  <h2>Models and data</h2>
  <p>Predictions are the mean of two graph neural networks trained on ${(68763).toLocaleString()} measured infinite-dilution activity coefficients of 336 solutes in 771 solvents between 250 and 556 K: a directed message-passing network (D-MPNN) trained on activity coefficients and on the differences between isomers, and the Gibbs–Helmholtz graph neural network (GH-GNN; Sanchez Medina and Minor, MIT licence). Each is averaged over three training runs. The figures above are the paper's validation results for this averaged prediction, so they are slightly better than the single-run figures in the paper.</p>
  <p>The ${st.solv.length.toLocaleString()} solvents are the ${nMeas} measured solvents, ${st.solv.filter((s) => s.src === 'ion combination').length.toLocaleString()} ionic liquids combined from the measured cations and anions, and ${st.solv.filter((s) => s.src === 'CHEM21' || s.src === 'Langner 2022').length} common organic solvents from the CHEM21 guide and Langner and Brabec (2022, CC BY 4.0). Ionic liquids containing aluminium, cobalt, iron or antimony are left out because GH-GNN cannot represent them.</p>
  <p>Everything runs in your browser; the structures you enter are not sent anywhere. The browser calculation reproduces the reference implementation to within 0.0002 in ln γ<sup>∞</sup>.</p>
  <h2>Citation</h2><p>If you use IsoScreen, please cite the accompanying paper, <i>Resolution limits of predicted selectivities in isomer separation</i> (submitted). Data, models and code: <a href="https://github.com/muyingjie71-oss/isoscreen" target="_blank" rel="noopener">https://github.com/muyingjie71-oss/isoscreen</a>.</p>`;
}
function maxHit() { let n = 0, k = 0; for (const p of st.pairs.values()) for (const m of p.m) { n++; if (Math.exp(Math.abs(m[2])) >= 1.05) k++; } return k / n; }

// ------------------------------------------------------------------ routing
function route() {
  const v = (location.hash || '#screen').slice(1); const view = ['screen', 'pair', 'method'].includes(v) ? v : 'screen';
  for (const n of ['screen', 'pair', 'method']) $('#v-' + n).classList.toggle('hidden', n !== view);
  document.querySelectorAll('nav a').forEach((a) => a.classList.toggle('on', a.dataset.view === view));
}
boot().catch((e) => { status(`<span class="err">${esc(e.message)}</span>`); console.error(e); });
