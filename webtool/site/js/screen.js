// Solvent-slice screener on onnxruntime-web: one instance per Web Worker (single-threaded WASM each; workers give parallelism
// without SharedArrayBuffer, so it runs on plain static hosting). Solute encodings come from engine.js on the main thread.

export function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

export class SliceScreener {
  // onnx: {name: ArrayBuffer} for dmpnn_prep_k, dmpnn_head_k, ghgnn_prep_k, ghgnn_head_k; slice: {lo, hi, dm: Float32Array(seeds*n*256), gh: Float32Array(seeds*n*226), hba, hbd}
  // enet: per seed {w1: Float32Array(32), b1: Float32Array(32), W2: Float32Array(32*51076), b2: Float32Array(51076)}
  static async create(ort, onnx, meta, slice, enet) {
    const s = new SliceScreener(); s.ort = ort; s.meta = meta; s.slice = slice; s.enet = enet; s.n = slice.hi - slice.lo; s.nk = meta.seeds.length; s.sess = {};
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    for (const [name, buf] of Object.entries(onnx)) s.sess[name] = await ort.InferenceSession.create(buf, opt);
    await s.prepare(); return s;
  }
  T(data, dims) { return new this.ort.Tensor('float32', data, dims); }
  async prepare() {
    const n = this.n; this.hs = []; this.sp = []; this.p1 = [];
    for (let k = 0; k < this.nk; k++) {
      const hs = this.slice.dm.subarray(k * n * 256, (k + 1) * n * 256); this.hs.push(hs);
      this.sp.push((await this.sess[`dmpnn_prep_${k}`].run({ hs: this.T(hs, [n, 256]) })).sp.data);
      const z = this.slice.gh.subarray(k * n * 226, (k + 1) * n * 226);
      this.p1.push((await this.sess[`ghgnn_prep_${k}`].run({ z: this.T(z, [n, 226]) })).p.data);
    }
    this.Wcache = Array.from({ length: this.nk }, () => new Map());
  }
  setEdgeWeights(list) { for (const [k, v, W] of list) this.Wcache[k].set(v, W); }   // [seed, inter_hb, Float32Array(226*226)] computed on the main thread
  edgeW(k, v) {
    const c = this.Wcache[k]; if (c.has(v)) return c.get(v);
    if (!this.enet) throw new Error('missing GH-GNN edge weight for inter_hb = ' + v);
    const { w1, b1, W2, b2 } = this.enet[k]; const D2 = b2.length; const out = Float32Array.from(b2); const h = new Float64Array(32);
    for (let j = 0; j < 32; j++) h[j] = Math.max(0, v * w1[j] + b1[j]);
    for (let j = 0; j < 32; j++) { const hj = h[j]; if (!hj) continue; const o = j * D2; for (let i = 0; i < D2; i++) out[i] += hj * W2[o + i]; }
    c.set(v, out); return out;
  }
  // solutes: [{dm: [Float32Array(256) per seed], gh: [Float32Array(226) per seed] | null, hba, hbd}]
  // returns per solute: {dm: Float32Array(seeds*n*K) ln g, A, B: Float32Array(seeds*n) (NaN if GH-GNN cannot featurise the solute)}
  async run(solutes, Tk, ghgnn = true) {
    const n = this.n, K = Tk.length, nk = this.nk, res = [];
    for (const so of solutes) {
      const dm = new Float32Array(nk * n * K), A = new Float32Array(nk * n).fill(NaN), B = new Float32Array(nk * n).fill(NaN);
      for (let k = 0; k < nk; k++) {
        const sc = this.meta.dmpnn[this.meta.seeds[k]].scalers;
        const tz = Float32Array.from(Tk, (T) => (Math.fround(1000 / T) - sc.t_mean) / sc.t_std);
        const o = await this.sess[`dmpnn_head_${k}`].run({ hi: this.T(so.dm[k], [1, 256]), hs: this.T(this.hs[k], [n, 256]), sp: this.T(this.sp[k], [n, 512]), tz: this.T(tz, [K]) });
        dm.set(o.lng.data, k * n * K);
        if (!ghgnn || !so.gh) continue;
        const p2 = (await this.sess[`ghgnn_prep_${k}`].run({ z: this.T(so.gh[k], [1, 226]) })).p.data;
        const groups = new Map();
        for (let i = 0; i < n; i++) { const v = Math.min(this.slice.hba[i], so.hbd) + Math.min(so.hba, this.slice.hbd[i]); if (!groups.has(v)) groups.set(v, []); groups.get(v).push(i); }
        for (const [v, idx] of groups) {
          const P = new Float32Array(idx.length * 226); idx.forEach((i, r) => P.set(this.p1[k].subarray(i * 226, (i + 1) * 226), r * 226));
          const g = await this.sess[`ghgnn_head_${k}`].run({ p1: this.T(P, [idx.length, 226]), p2: this.T(p2, [1, 226]), Wv: this.T(this.edgeW(k, v), [226, 226]) });
          idx.forEach((i, r) => { A[k * n + i] = g.A.data[r]; B[k * n + i] = g.B.data[r]; });
        }
      }
      res.push({ dm, A, B });
    }
    return res;
  }
}
