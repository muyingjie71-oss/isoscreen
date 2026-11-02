// IsoScreen browser featuriser: RDKit.js (MinimalLib) -> the exact molecular graphs and descriptors the two models were trained on.
// D-MPNN: Layer A graphs.graph() (50 atom + 14 bond features). GH-GNN: mol2graph atom/bond features (37 + 9) and
// APol, BPol (mordred, polarizability94, explicit H), TopoPSA (= RDKit TPSA, N/O only), NumHBA, NumHBD.
// RDKit.js does not expose bond conjugation; it is reproduced from RDKit's ConjugHybrid.cpp / Aromaticity.cpp (Release_2025_09_4).
// Hybridisation, valence and ring membership are read from RDKit itself through SMARTS queries.
// Every feature is checked against the Python featurisers on 7,892 molecules (tests/featurize_test.mjs).

export function makeFeaturizer(RDKit, ELEMENTS) {
  const EL = ELEMENTS;
  const qcache = new Map();
  const qmol = (s) => { if (!qcache.has(s)) qcache.set(s, RDKit.get_qmol(s)); return qcache.get(s); };
  const matches = (mol, smarts) => { const r = JSON.parse(mol.get_substruct_matches(qmol(smarts))); return Array.isArray(r) ? r : []; };
  const atomSet = (mol, smarts) => new Set(matches(mol, smarts).map((m) => m.atoms[0]));
  const bondSet = (mol, smarts) => new Set(matches(mol, smarts).map((m) => m.bonds[0]));

  function parse(smiles) {
    const mol = RDKit.get_mol(smiles);
    if (!mol || !mol.is_valid()) { if (mol) mol.delete(); return null; }
    try {
      const J = JSON.parse(mol.get_json()); const D = J.defaults; const M = J.molecules[0];
      const ext = (M.extensions || []).find((e) => e.name === 'rdkitRepresentation') || {};
      const n = M.atoms.length;
      const atoms = M.atoms.map((a, i) => ({ i, z: a.z ?? D.atom.z, hs: a.impHs ?? D.atom.impHs, chg: a.chg ?? D.atom.chg,
        nRad: a.nRad ?? D.atom.nRad, iso: a.isotope ?? D.atom.isotope, stereo: a.stereo ?? D.atom.stereo, nbrs: [], bonds: [] }));
      const bonds = (M.bonds || []).map((b, k) => ({ k, a: b.atoms[0], b: b.atoms[1], bo: b.bo ?? D.bond.bo, stereo: b.stereo ?? D.bond.stereo, stereoAtoms: b.stereoAtoms || null }));
      for (const b of bonds) { atoms[b.a].nbrs.push(b.b); atoms[b.b].nbrs.push(b.a); atoms[b.a].bonds.push(b.k); atoms[b.b].bonds.push(b.k); }
      const aromA = new Set(ext.aromaticAtoms || []); const aromB = bondSet(mol, '*:*');  // RDKit aromatic bond flag (JSON aromaticBonds can include non-aromatic bonds between aromatic atoms)
      const cipRanks = ext.cipRanks || null; const cipCodes = new Map((ext.cipCodes || []).map(([i, c]) => [i, c]));
      // RDKit-perceived properties through SMARTS
      const hyb = new Array(n).fill('OTHER');
      for (const [code, name] of [[0, 'S'], [1, 'SP'], [2, 'SP2'], [3, 'SP3'], [4, 'SP3D'], [5, 'SP3D2']]) for (const i of atomSet(mol, `[^${code}]`)) hyb[i] = name;
      const val = new Array(n).fill(null);
      for (let v = 0; v <= 12; v++) for (const i of atomSet(mol, `[v${v}]`)) val[i] = v;
      const ringA = atomSet(mol, '[R]'); const ringB = bondSet(mol, '*@*');
      for (const a of atoms) {
        a.arom = aromA.has(a.i); a.inRing = ringA.has(a.i); a.hyb = hyb[a.i]; a.totalValence = val[a.i];
        a.degree = a.nbrs.length; a.totalDegree = a.degree + a.hs; a.cip = cipCodes.get(a.i) || null;
      }
      for (const b of bonds) { b.arom = aromB.has(b.k); b.inRing = ringB.has(b.k); b.type = b.arom ? 'AROMATIC' : ({ 1: 'SINGLE', 2: 'DOUBLE', 3: 'TRIPLE' }[b.bo] || 'OTHER'); }
      setConjugation(atoms, bonds);
      for (const b of bonds) b.ez = bondEZ(b, atoms, cipRanks);
      const desc = JSON.parse(mol.get_descriptors());
      return { smiles, atoms, bonds, desc, inchikey: (() => { try { return RDKit.get_inchikey_for_inchi(mol.get_inchi()); } catch (e) { return null; } })(),
               canonical: mol.get_smiles() };
    } finally { mol.delete(); }
  }

  // ---- conjugation (RDKit MolOps::setConjugation) ----
  function contrib(b) { return b.type === 'AROMATIC' ? 1.5 : ({ SINGLE: 1, DOUBLE: 2, TRIPLE: 3 }[b.type] ?? 0); }
  function countAtomElec(a, bonds) {
    const e = EL[a.z]; const dv = e.dv;
    if (dv <= 1) return -1;
    let degree = a.degree + a.hs;
    for (const k of a.bonds) if (!contrib(bonds[k])) degree -= 1;
    if (degree > 3) return -1;
    let nlp = e.nouter - dv; nlp = Math.max(nlp - a.chg, 0);
    let res = (dv - degree) + nlp - a.nRad;
    // RDKit then lowers res to 1 when res > 1 and the atom has more than one unsaturation; only the sign of res is used here
    return res;
  }
  function isConjCand(a, bonds) {
    const e = EL[a.z];
    if (!a.chg && e.val0 >= 0 && a.totalValence > e.val0) return false;
    const nouter = e.nouter;
    return ((a.z <= 10) || (nouter !== 5 && nouter !== 6) || (nouter === 6 && a.totalDegree < 2)) && countAtomElec(a, bonds) > 0;
  }
  function setConjugation(atoms, bonds) {
    for (const b of bonds) b.conj = b.arom;
    const cand = atoms.map((a) => isConjCand(a, bonds));
    for (const a of atoms) {
      if (!cand[a.i]) continue;
      const sbo = a.degree + a.hs; if (sbo < 2 || sbo > 3) continue;
      for (const k1 of a.bonds) {
        const b1 = bonds[k1]; const o1 = b1.a === a.i ? b1.b : b1.a;
        if (contrib(b1) < 1.5 || !cand[o1]) continue;
        for (const k2 of a.bonds) {
          if (k1 === k2) continue;
          const b2 = bonds[k2]; const o2 = b2.a === a.i ? b2.b : b2.a; const at2 = atoms[o2];
          if (at2.degree + at2.hs > 3) continue;
          if (cand[o2]) { b1.conj = true; b2.conj = true; }
        }
      }
    }
  }
  // ---- double-bond E/Z from cis/trans + legacy CIP ranks ----
  function bondEZ(b, atoms, ranks) {
    if (b.type !== 'DOUBLE' || !b.stereoAtoms || (b.stereo !== 'cis' && b.stereo !== 'trans')) return 'NONE';
    if (!ranks) return 'UNKNOWN';
    const top = (end, other) => { let best = null; for (const nb of atoms[end].nbrs) { if (nb === other) continue; if (best === null || ranks[nb] > ranks[best]) best = nb; } return best; };
    const [s1, s2] = b.stereoAtoms; const t1 = top(b.a, b.b), t2 = top(b.b, b.a);
    let same = b.stereo === 'cis'; if ((s1 !== t1) !== (s2 !== t2)) same = !same;
    return same ? 'Z' : 'E';
  }

  // ---- D-MPNN Layer A features ----
  const D_EL = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53];
  const oh = (x, vals) => { const r = vals.map((v) => (x === v ? 1 : 0)); r.push(vals.includes(x) ? 0 : 1); return r; };
  function dmpnnGraph(P) {
    const atom = P.atoms.map((a) => {
      if (a.iso) throw new Error('isotope-labelled atoms are not supported');
      return [...oh(a.z, D_EL), ...oh(a.totalDegree, [0, 1, 2, 3, 4, 5, 6]), ...oh(a.chg, [-3, -2, -1, 0, 1, 2, 3]), ...oh(a.hs, [0, 1, 2, 3, 4]),
        ...oh(a.hyb, ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']), a.arom ? 1 : 0, Math.fround(EL[a.z].mass / 100),
        ...oh(a.stereo === 'cw' ? 'CW' : a.stereo === 'ccw' ? 'CCW' : a.stereo === 'unspecified' ? 'UNSPEC' : 'OTHER', ['UNSPEC', 'CW', 'CCW']),
        ...oh(a.cip, ['R', 'S'])];
    });
    const src = [], dst = [], rev = [], bond = [];
    for (const b of P.bonds) {
      const st = b.ez === 'E' ? 'E' : b.ez === 'Z' ? 'Z' : b.ez === 'NONE' ? 'NONE' : 'OTHER';
      const f = [...oh(b.type, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']), b.conj ? 1 : 0, b.inRing ? 1 : 0, ...oh(st, ['NONE', 'ANY', 'Z', 'E', 'CIS', 'TRANS'])];
      const e = src.length; src.push(b.a, b.b); dst.push(b.b, b.a); rev.push(e + 1, e); bond.push(f, f);
    }
    return { atom, bond, src, dst, rev };
  }

  // ---- GH-GNN features ----
  const G_EL = ['C', 'N', 'O', 'Cl', 'S', 'F', 'Br', 'I', 'Si', 'Sn', 'Pb', 'Ge', 'H', 'P', 'Hg', 'Te', 'B'];
  const strict = (x, vals, what) => { const i = vals.indexOf(x); if (i < 0) throw new Error(`outside the GH-GNN featuriser: ${what} ${x}`); return vals.map((_, j) => (j === i ? 1 : 0)); };
  function ghgnnGraph(P) {
    const x = P.atoms.map((a) => [...strict(EL[a.z].sym, G_EL, 'element'), a.inRing ? 1 : 0, a.arom ? 1 : 0,
      ...strict(a.hyb, ['S', 'SP3D2', 'SP', 'SP2', 'SP3'], 'hybridisation'), ...strict(a.degree, [0, 1, 2, 3, 4, 5, 6], 'degree'),
      ...strict(a.chg, [0, 1, -1, 2, -2, 3, -3], 'charge'), ...strict(a.hs, [0, 1, 2, 3], 'H count'),
      ...strict(a.stereo === 'cw' ? 'CW' : a.stereo === 'ccw' ? 'CCW' : a.stereo === 'unspecified' ? 'UNSPEC' : 'OTHER', ['UNSPEC', 'CW', 'CCW'], 'chirality')]);
    const row = [], col = [], ea = [];
    for (const b of P.bonds) {
      const f = [b.type === 'SINGLE' ? 1 : 0, b.type === 'DOUBLE' ? 1 : 0, b.type === 'TRIPLE' ? 1 : 0, b.type === 'AROMATIC' ? 1 : 0, b.conj ? 1 : 0, b.inRing ? 1 : 0,
        ...strict(b.ez, ['NONE', 'Z', 'E'], 'bond stereo')];
      row.push(b.a, b.b); col.push(b.b, b.a); ea.push(f, f);
    }
    // mordred APol / BPol with explicit hydrogens (polarizability94), TopoPSA (N, O) = RDKit TPSA, HBA / HBD = rdMolDescriptors
    const pol = (z) => EL[z].pol94; let ap = 0, bp = 0;
    for (const a of P.atoms) { ap += pol(a.z); for (let h = 0; h < a.hs; h++) { ap += pol(1); bp += Math.abs(pol(a.z) - pol(1)); } }
    for (const b of P.bonds) bp += Math.abs(pol(P.atoms[b.a].z) - pol(P.atoms[b.b].z));
    return { x, ea, row, col, ap, bp, tpsa: P.desc.tpsa, hba: P.desc.NumHBA, hbd: P.desc.NumHBD };
  }
  return { parse, dmpnnGraph, ghgnnGraph };
}
