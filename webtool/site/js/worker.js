// One Web Worker = one slice of the solvent library on its own single-threaded onnxruntime-web instance.
import * as ort from '../vendor/ort.wasm.bundle.min.js';
import { SliceScreener } from './screen.js';
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { mjs: new URL('../vendor/ort-wasm-simd-threaded.js', import.meta.url).href, wasm: new URL('../vendor/ort-wasm-simd-threaded.wasm', import.meta.url).href };
let S = null;
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      if (m.ortWasm) ort.env.wasm.wasmBinary = new Uint8Array(m.ortWasm);   // binary handed over by the page: no second download, no MIME issues
      const onnx = {}; for (const [k, buf] of Object.entries(m.onnx)) onnx[k] = new Uint8Array(buf);
      S = await SliceScreener.create(ort, onnx, m.meta, m.slice, null);
      self.postMessage({ type: 'ready', id: m.id, n: S.n });
    } else if (m.type === 'run') {
      S.setEdgeWeights(m.edge || []);
      const res = await S.run(m.solutes, m.T, m.ghgnn);
      const transfer = res.flatMap((r) => [r.dm.buffer, r.A.buffer, r.B.buffer]);
      self.postMessage({ type: 'result', id: m.id, res }, transfer);
    }
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) }); }
};
