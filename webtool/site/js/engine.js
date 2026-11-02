// IsoScreen inference engine (plain JS, runs in Web Workers or Node). Reproduces the split models of engine/split_models.py:
// solute encoders run once per query; pair heads run for every solvent. Weights float32, arithmetic float64.
// Verified against the Python production predictors on the 5,040-point check grid (tests/engine_test.mjs).

const silu = (x) => x / (1 + Math.exp(-x));
const relu = (x) => (x > 0 ? x : 0);
const sigm = (x) => 1 / (1 + Math.exp(-x));

export function loadWeights(meta, buf) {
  const W = {};
  for (const [name, [off, shape]] of Object.entries(meta.index)) {
    const n = shape.reduce((a, b) => a * b, 1);
    W[name] = { d: buf.subarray(off, off + n), shape };
  }
  return W;
}

// y (len cols) = b + x (len rows) @ M (rows x cols)
function vm(x, M, b, out) {
  const [rows, cols] = M.shape; const d = M.d;
  if (b) for (let j = 0; j < cols; j++) out[j] = b.d[j]; else out.fill(0, 0, cols);
  for (let i = 0; i < rows; i++) { const xi = x[i]; if (xi === 0) continue; const o = i * cols; for (let j = 0; j < cols; j++) out[j] += xi * d[o + j]; }
  return out;
}
// block version: X is B x rows (row stride xs), OUT is B x cols, accumulate on top of init (if given)
function bm(X, B, xs, xo, M, OUT, os, oo) {
  const [rows, cols] = M.shape; const d = M.d;
  for (let i = 0; i < rows; i++) {
    const o = i * cols;
    for (let b = 0; b < B; b++) { const xi = X[xo + b * xs + i]; if (xi === 0) continue; const ob = oo + b * os; for (let j = 0; j < cols; j++) OUT[ob + j] += xi * d[o + j]; }
  }
}
const F64 = (n) => new Float64Array(n);

// ------------------------------------------------------------------ solute encoders
export function dmpnnEncode(W, k, g, depth = 4) {
  const H = 256, N = g.atom.length, E = g.src.length, P = `d${k}/`;
  const inBond = F64(64), tmp = F64(H);
  const h0 = Array.from({ length: E }, () => F64(H));
  for (let e = 0; e < E; e++) {
    const a = g.atom[g.src[e]], bf = g.bond[e];
    for (let i = 0; i < 50; i++) inBond[i] = a[i]; for (let i = 0; i < 14; i++) inBond[50 + i] = bf[i];
    vm(inBond, W[P + 'bond_input/W'], W[P + 'bond_input/b'], h0[e]); for (let j = 0; j < H; j++) h0[e][j] = silu(h0[e][j]);
  }
  let h = h0.map((v) => v.slice());
  const incoming = () => { const inc = Array.from({ length: N }, () => F64(H)); for (let e = 0; e < E; e++) { const t = inc[g.dst[e]], s = h[e]; for (let j = 0; j < H; j++) t[j] += s[j]; } return inc; };
  for (let it = 0; it < depth - 1; it++) {
    const inc = incoming(); const nh = [];
    for (let e = 0; e < E; e++) {
      const a = inc[g.src[e]], r = h[g.rev[e]]; const msg = F64(H); for (let j = 0; j < H; j++) msg[j] = a[j] - r[j];
      vm(msg, W[P + 'bond_message/W'], null, tmp); const o = F64(H); for (let j = 0; j < H; j++) o[j] = silu(h0[e][j] + tmp[j]); nh.push(o);
    }
    h = nh;
  }
  const inc = incoming(); const pooled = F64(H); const inA = F64(306);
  for (let n = 0; n < N; n++) {
    const a = g.atom[n]; for (let i = 0; i < 50; i++) inA[i] = a[i]; for (let j = 0; j < H; j++) inA[50 + j] = inc[n][j];
    vm(inA, W[P + 'atom_output/W'], W[P + 'atom_output/b'], tmp); for (let j = 0; j < H; j++) pooled[j] += silu(tmp[j]);
  }
  const f = F64(513); const cnt = Math.max(N, 1);
  for (let j = 0; j < H; j++) { f[j] = pooled[j] / cnt; f[H + j] = pooled[j]; } f[512] = Math.log1p(N);
  const out = vm(f, W[P + 'size_aware_readout/W'], W[P + 'size_aware_readout/b'], F64(H)); for (let j = 0; j < H; j++) out[j] = silu(out[j]);
  return out;
}

export function ghgnnEncode(W, k, g, eps) {
  const P = `g${k}/`; const N = g.x.length, E = g.row.length;
  const mlp = (x, pre) => { const h = vm(x, W[pre + '1/W'], W[pre + '1/b'], F64(W[pre + '1/W'].shape[1])); for (let j = 0; j < h.length; j++) h[j] = relu(h[j]); return vm(h, W[pre + '2/W'], W[pre + '2/b'], F64(W[pre + '2/W'].shape[1])); };
  const cat = (...vs) => { const n = vs.reduce((s, v) => s + v.length, 0); const o = F64(n); let p = 0; for (const v of vs) { o.set(v, p); p += v.length; } return o; };
  const meta = (gname, x, ea, u) => {
    const pre = P + gname + '/';
    const ea2 = []; for (let e = 0; e < E; e++) ea2.push(mlp(cat(x[g.row[e]], x[g.col[e]], ea[e], u), pre + 'e'));
    const dh = ea2.length ? ea2[0].length : W[pre + 'e2/W'].shape[1];
    const agg = Array.from({ length: N }, () => F64(dh)); for (let e = 0; e < E; e++) { const t = agg[g.col[e]]; for (let j = 0; j < dh; j++) t[j] += ea2[e][j]; }
    const x2 = []; for (let n = 0; n < N; n++) x2.push(mlp(cat(x[n], agg[n], u), pre + 'n'));
    const dx = x2[0].length; const nm = F64(dx); for (const v of x2) for (let j = 0; j < dx; j++) nm[j] += v[j] / N;
    const em = F64(dh); if (E) for (const v of ea2) for (let j = 0; j < dh; j++) em[j] += v[j] / E;
    return [x2, ea2, mlp(cat(u, nm, em), pre + 'g')];
  };
  const norm = (name, x) => {
    const w = W[P + name + '/w'].d, b = W[P + name + '/b'].d, ms = W[P + name + '/ms'].d, D = x[0].length;
    const mean = F64(D); for (const v of x) for (let j = 0; j < D; j++) mean[j] += v[j] / N;
    const out = x.map((v) => { const o = F64(D); for (let j = 0; j < D; j++) o[j] = v[j] - mean[j] * ms[j]; return o; });
    const vr = F64(D); for (const o of out) for (let j = 0; j < D; j++) vr[j] += (o[j] * o[j]) / N;
    for (const o of out) for (let j = 0; j < D; j++) o[j] = (w[j] * o[j]) / Math.sqrt(vr[j] + eps) + b[j];
    return out;
  };
  let x = g.x.map((r) => Float64Array.from(r)), ea = g.ea.map((r) => Float64Array.from(r)), u = Float64Array.from([g.ap, g.bp, g.tpsa]);
  [x, ea, u] = meta('graphnet1', x, ea, u); x = norm('gnorm1', x);
  [x, ea, u] = meta('graphnet2', x, ea, u); x = norm('gnorm2', x);
  const D = x[0].length; const xg = F64(D); for (const v of x) for (let j = 0; j < D; j++) xg[j] += v[j] / N;
  return cat(xg, u);
}

// ------------------------------------------------------------------ solvent-side preparation (once per page load, per worker slice)
export function prepareSolvents(W, meta, dmEmb, ghEmb, S, lo, hi) {
  const nS = hi - lo, out = { lo, hi, dm: [], gh: [] }, BL = 64;
  for (let k = 0; k < meta.seeds.length; k++) {
    const P = `d${k}/`, Ws = W[P + 'f1/Ws'], Wd = W[P + 'f1/Wd'];
    const hs = Float32Array.from(dmEmb.subarray((k * S + lo) * 256, (k * S + hi) * 256));
    const sp = new Float32Array(nS * 512), A = F64(BL * 512), Bd = F64(BL * 512), X = F64(BL * 256);
    for (let s0 = 0; s0 < nS; s0 += BL) {
      const B = Math.min(BL, nS - s0); A.fill(0); Bd.fill(0);
      for (let i = 0; i < B * 256; i++) X[i] = hs[s0 * 256 + i];
      bm(X, B, 256, 0, Ws, A, 512, 0); bm(X, B, 256, 0, Wd, Bd, 512, 0);
      for (let i = 0; i < B * 512; i++) sp[s0 * 512 + i] = A[i] - Bd[i];
    }
    out.dm.push({ hs, sp });
    const G = `g${k}/`, Wp = W[G + 'proj/W'], bp = W[G + 'proj/b'].d, p1 = new Float32Array(nS * 226), Z = F64(BL * 226), O = F64(BL * 226);
    for (let s0 = 0; s0 < nS; s0 += BL) {
      const B = Math.min(BL, nS - s0);
      for (let i = 0; i < B * 226; i++) Z[i] = ghEmb[(k * S + lo + s0) * 226 + i];
      for (let b = 0; b < B; b++) for (let j = 0; j < 226; j++) O[b * 226 + j] = bp[j];
      bm(Z, B, 226, 0, Wp, O, 226, 0);
      for (let i = 0; i < B * 226; i++) p1[s0 * 226 + i] = relu(O[i]);
    }
    out.gh.push({ p1 });
  }
  return out;
}

// ------------------------------------------------------------------ pair heads for a solvent slice
// solutes: [{dm:[Float64Array(256) x seeds], gh:[Float64Array(226) x seeds] | null, hba, hbd}], T: [K kelvin]
// returns dm: Float32Array(nSolute * seeds * nS * K) ln g ; ghA, ghB: Float32Array(nSolute * seeds * nS)
export function screenSlice(W, meta, prep, solv, solutes, T) {
  const nS = prep.hi - prep.lo, K = T.length, nk = meta.seeds.length, nu = solutes.length, BL = 32;
  const dmOut = new Float32Array(nu * nk * nS * K), ghA = new Float32Array(nu * nk * nS).fill(NaN), ghB = new Float32Array(nu * nk * nS).fill(NaN);
  for (let k = 0; k < nk; k++) {
    // ---------------- D-MPNN
    const P = `d${k}/`, sc = meta.dmpnn[meta.seeds[k]].scalers;
    const Wi = W[P + 'f1/Wi'], Wd = W[P + 'f1/Wd'], Wp = W[P + 'f1/Wp'], Wt = W[P + 'f1/Wt'], b1 = W[P + 'f1/b'], W2 = W[P + 'f2/W'], b2 = W[P + 'f2/b'].d, w3 = W[P + 'f3/W'].d, b3 = W[P + 'f3/b'].d[0];
    const htW = T.map((TK) => { const tz = (Math.fround(1000 / TK) - sc.t_mean) / sc.t_std; const ht = F64(32); const tw = W[P + 'temp/W'].d, tb = W[P + 'temp/b'].d; for (let j = 0; j < 32; j++) ht[j] = silu(Math.fround(tz) * tw[j] + tb[j]); return vm(ht, Wt, null, F64(512)); });
    const { hs, sp } = prep.dm[k];
    for (let u = 0; u < nu; u++) {
      const hiv = solutes[u].dm[k]; const c = vm(hiv, Wi, b1, F64(512)), cd = vm(hiv, Wd, null, F64(512)); for (let j = 0; j < 512; j++) c[j] += cd[j];
      const prod = F64(BL * 256), base = F64(BL * 512), x = F64(BL * 512), y = F64(BL * 256);
      for (let s0 = 0; s0 < nS; s0 += BL) {
        const B = Math.min(BL, nS - s0);
        for (let b = 0; b < B; b++) { const s = s0 + b; for (let j = 0; j < 256; j++) prod[b * 256 + j] = hiv[j] * hs[s * 256 + j]; for (let j = 0; j < 512; j++) base[b * 512 + j] = c[j] + sp[s * 512 + j]; }
        bm(prod, B, 256, 0, Wp, base, 512, 0);
        for (let t = 0; t < K; t++) {
          const hw = htW[t];
          for (let b = 0; b < B; b++) for (let j = 0; j < 512; j++) x[b * 512 + j] = silu(base[b * 512 + j] + hw[j]);
          for (let b = 0; b < B; b++) for (let j = 0; j < 256; j++) y[b * 256 + j] = b2[j];
          bm(x, B, 512, 0, W2, y, 256, 0);
          for (let b = 0; b < B; b++) { let o = b3; for (let j = 0; j < 256; j++) o += silu(y[b * 256 + j]) * w3[j]; dmOut[((u * nk + k) * nS + s0 + b) * K + t] = o * sc.y_std + sc.y_mean; }
        }
      }
    }
    // ---------------- GH-GNN
    const ghSol = solutes.map((x) => x.gh); if (ghSol.every((g) => !g)) continue;
    const G = `g${k}/`, D = 226, Wroot = W[G + 'root/W'], bconv = W[G + 'conv/b'].d, Wih = W[G + 'gru/Wih'], Whh = W[G + 'gru/Whh'], bih = W[G + 'gru/bih'], bhh = W[G + 'gru/bhh'];
    const edgeW = new Map(); const getW = (v) => { if (!edgeW.has(v)) { const h = F64(32), w1 = W[G + 'enet1/W'].d, eb = W[G + 'enet1/b'].d; for (let j = 0; j < 32; j++) h[j] = relu(v * w1[j] + eb[j]); edgeW.set(v, { d: Float32Array.from(vm(h, W[G + 'enet2/W'], W[G + 'enet2/b'], F64(D * D))), shape: [D, D] }); } return edgeW.get(v); };
    const sol = solutes.map((x) => { if (!x.gh) return null; const p = vm(x.gh[k], W[G + 'proj/W'], W[G + 'proj/b'], F64(D)); for (let j = 0; j < D; j++) p[j] = relu(p[j]); return { p, r: vm(p, Wroot, null, F64(D)), gh: vm(p, Whh, bhh, F64(3 * D)), q: new Map(), hba: x.hba, hbd: x.hbd }; });
    const { p1 } = prep.gh[k]; const GB = 32;
    const heads = ['a', 'b'].map((h) => [1, 2, 3].map((i) => [W[`${G}mlp${i}${h}/W`], W[`${G}mlp${i}${h}/b`].d]));
    const [Cmin, Cmax] = meta.ghgnn.C, [Dmin, Dmax] = meta.ghgnn.D, T1 = meta.ghgnn.T1, T2 = meta.ghgnn.T2;
    const P1 = F64(GB * D), R1 = F64(GB * D), G1 = F64(GB * 3 * D), M = F64(GB * D), Cc = F64(GB * D), GI = F64(GB * 3 * D), XG = F64(GB * 2 * D), H1 = F64(GB * 113), H2 = F64(GB * 113), Pg = F64(GB * D), Mg = F64(GB * D);
    const fillBias = (O, B, n, bias) => { for (let b = 0; b < B; b++) for (let j = 0; j < n; j++) O[b * n + j] = bias[j]; };
    const gruB = (B, gh, ghStride, h, hStride, off) => { for (let b = 0; b < B; b++) { const go = b * ghStride, ho = b * hStride; for (let j = 0; j < D; j++) {
      const gi = GI[b * 3 * D + j], gz = GI[b * 3 * D + D + j], gn = GI[b * 3 * D + 2 * D + j];
      const r = sigm(gi + gh[go + j]), z = sigm(gz + gh[go + D + j]), n = Math.tanh(gn + r * gh[go + 2 * D + j]); XG[b * 2 * D + off + j] = (1 - z) * n + z * h[ho + j]; } } };
    for (let s0 = 0; s0 < nS; s0 += GB) {
      const B = Math.min(GB, nS - s0);
      for (let i = 0; i < B * D; i++) P1[i] = p1[s0 * D + i];
      R1.fill(0, 0, B * D); bm(P1, B, D, 0, Wroot, R1, D, 0);
      fillBias(G1, B, 3 * D, bhh.d); bm(P1, B, D, 0, Whh, G1, 3 * D, 0);
      for (let u = 0; u < nu; u++) {
        const so = sol[u]; if (!so) continue;
        // m = (p1 + p2) @ W(v), v = inter_hb; solvents of the block grouped by v
        const groups = new Map();
        for (let b = 0; b < B; b++) { const sv = solv[prep.lo + s0 + b]; const v = Math.min(sv.hba, so.hbd) + Math.min(so.hba, sv.hbd); if (!groups.has(v)) groups.set(v, []); groups.get(v).push(b); }
        for (const [v, idx] of groups) {
          const Wv = getW(v); if (!so.q.has(v)) so.q.set(v, vm(so.p, Wv, null, F64(D))); const q = so.q.get(v);
          idx.forEach((b, r) => { for (let j = 0; j < D; j++) { Pg[r * D + j] = P1[b * D + j]; Mg[r * D + j] = q[j]; } });
          bm(Pg, idx.length, D, 0, Wv, Mg, D, 0);
          idx.forEach((b, r) => { for (let j = 0; j < D; j++) M[b * D + j] = Mg[r * D + j]; });
        }
        for (let b = 0; b < B; b++) for (let j = 0; j < D; j++) Cc[b * D + j] = relu(M[b * D + j] + R1[b * D + j] + bconv[j]);
        fillBias(GI, B, 3 * D, bih.d); bm(Cc, B, D, 0, Wih, GI, 3 * D, 0); gruB(B, G1, 3 * D, P1, D, 0);
        for (let b = 0; b < B; b++) for (let j = 0; j < D; j++) Cc[b * D + j] = relu(M[b * D + j] + so.r[j] + bconv[j]);
        fillBias(GI, B, 3 * D, bih.d); bm(Cc, B, D, 0, Wih, GI, 3 * D, 0); gruB(B, so.gh, 0, so.p, 0, D);
        const o = heads.map(([[W1, B1], [W2h, B2], [W3, B3]]) => {
          fillBias(H1, B, 113, B1); bm(XG, B, 2 * D, 0, W1, H1, 113, 0); for (let i = 0; i < B * 113; i++) H1[i] = relu(H1[i]);
          fillBias(H2, B, 113, B2); bm(H1, B, 113, 0, W2h, H2, 113, 0);
          const r = F64(B); for (let b = 0; b < B; b++) { let x = B3[0]; for (let j = 0; j < 113; j++) x += relu(H2[b * 113 + j]) * W3.d[j]; r[b] = x; } return r; });
        for (let b = 0; b < B; b++) {
          const Cv = o[0][b] * (Cmax - Cmin) + Cmin, Dv = o[1][b] * (Dmax - Dmin) + Dmin; const Bv = Dv * 2 * T1 * T2 / (T2 - T1), Av = Cv - Bv * (T1 + T2) / (2 * T1 * T2);
          ghA[(u * nk + k) * nS + s0 + b] = Av; ghB[(u * nk + k) * nS + s0 + b] = Bv;
        }
      }
    }
  }
  return { dm: dmOut, ghA, ghB, nS, K, nk, nu };
}
