// Isomer type of a pair that is not in the dataset (dataset pairs use the curated labels).
// Port of data_build/isomer_type.py, which agrees with the curated labels for 357 of the 359 dataset pairs.
const FG = {
  alcohol: '[CX4][OX2H]', phenol: 'c[OX2H]', ether: '[OD2]([#6])[#6]', aldehyde: '[CX3H1](=O)[#6,#1]', ketone: '[#6][CX3](=O)[#6]', acid: 'C(=O)[OH]',
  ester: '[CX3](=O)[OX2][#6]', amine1: '[NX3H2;!$(N-C=O);!$(N-a)]', amine2: '[NX3H1;!$(N-C=O);!$(N-a)]', amine3: '[NX3H0;!$(N-C=O);!$(N-a)]',
  aniline: '[NX3]a', amide: 'NC=O', nitrile: 'C#N', alkene: 'C=C', alkyne: 'C#C', aromatic: 'a', halogen: '[F,Cl,Br,I]', thiol: '[SX2H]',
  sulfide: '[SX2]([#6])[#6]', nitro: '[N+](=O)[O-]', carbonate: 'O=C(O)O',
};

export function makeIsoType(RDKit) {
  const q = new Map(); const qmol = (s) => { if (!q.has(s)) q.set(s, RDKit.get_qmol(s)); return q.get(s); };
  function info(smiles) {
    const mol = RDKit.get_mol(smiles);
    try {
      const J = JSON.parse(mol.get_json()); const M = J.molecules[0]; const D = J.defaults;
      const ext = (M.extensions || []).find((e) => e.name === 'rdkitRepresentation') || {};
      const atoms = M.atoms.map((a) => ({ z: a.z ?? D.atom.z, nb: [] })); const bonds = (M.bonds || []).map((b) => b.atoms);
      for (const [x, y] of bonds) { atoms[x].nb.push(y); atoms[y].nb.push(x); }
      const rings = ext.atomRings || []; const inRing = new Set(rings.flat());
      const fg = Object.values(FG).map((s) => { const r = JSON.parse(mol.get_substruct_matches(qmol(s))); return Array.isArray(r) ? r.length : 0; });
      return { atoms, rings, inRing, fg };
    } finally { mol.delete(); }
  }
  // canonical SMILES of the forest left after keeping only atoms in `keep` (all carbon, single bonds); `mark` atoms written as [13C]
  function forest(I, keep, mark) {
    const seen = new Set(), parts = [];
    const dfs = (i, from) => { seen.add(i); let s = mark.has(i) ? '[13C]' : 'C'; const ch = I.atoms[i].nb.filter((j) => j !== from && keep.has(j) && !seen.has(j));
      ch.forEach((j, k) => { const sub = dfs(j, i); s += k < ch.length - 1 ? `(${sub})` : sub; }); return s; };
    for (const i of keep) if (!seen.has(i)) parts.push(dfs(i, -1));
    const can = parts.map((p) => { const m = RDKit.get_mol(p); const c = m.get_smiles(); m.delete(); return c; });
    return can.sort().join('.');
  }
  const idx = (I) => I.atoms.map((_, i) => i);
  const cskel = (I) => forest(I, new Set(idx(I).filter((i) => I.atoms[i].z === 6)), new Set());
  const subst = (I) => { const keep = new Set(idx(I).filter((i) => !I.inRing.has(i))); const mark = new Set([...keep].filter((i) => I.atoms[i].nb.some((j) => I.inRing.has(j)))); return forest(I, keep, mark); };
  const ringSizes = (I) => I.rings.map((r) => r.length).sort((a, b) => a - b).join(',');
  return function classify(sa, sb) {
    const a = info(sa), b = info(sb); const ra = a.rings.length, rb = b.rings.length;
    if (ra !== rb) return Math.min(ra, rb) === 0 ? 'ring-chain' : 'skeletal (ring system)';
    if (a.fg.join() !== b.fg.join()) {
      const names = Object.keys(FG); const diff = names.filter((_, i) => a.fg[i] !== b.fg[i]);
      return diff.every((d) => d === 'alkene' || d === 'alkyne') ? 'unsaturation type' : 'functional group';
    }
    if (ringSizes(a) !== ringSizes(b)) return 'skeletal (ring system)';
    const heteroInChain = a.atoms.some((at, i) => at.z !== 6 && !a.inRing.has(i) && at.nb.filter((j) => a.atoms[j].z === 6).length >= 2);
    if (heteroInChain) return 'connectivity (metamer)';
    if (ra) return subst(a) === subst(b) ? 'positional' : 'skeletal (chain branching)';
    return cskel(a) === cskel(b) ? 'positional' : 'skeletal (chain branching)';
  };
}
