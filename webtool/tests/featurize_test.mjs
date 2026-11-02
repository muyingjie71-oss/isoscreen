// Compare the browser featuriser with the Python reference, molecule by molecule.
import fs from 'fs'; import readline from 'readline'; import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const initRDKit = require('@rdkit/rdkit');
const { makeFeaturizer } = await import('../site/js/featurize.js');
const R = await initRDKit(); const EL = JSON.parse(fs.readFileSync('../site/js/elements.json'));
const F = makeFeaturizer(R, EL);
const eq = (a, b, tol = 0) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i];
  if (Array.isArray(x)) { if (!eq(x, y, tol)) return false; } else if (Math.abs(x - y) > tol) return false; } return true; };
const rl = readline.createInterface({ input: fs.createReadStream(process.argv[2] || 'ref_features.jsonl') });
let n = 0, okD = 0, okG = 0, nG = 0, gUnsupJS = 0, gUnsupPy = 0; const bad = [];
for await (const line of rl) {
  const r = JSON.parse(line); n++;
  const P = F.parse(r.smiles); if (!P) { bad.push(['parse', r.smiles]); continue; }
  try {
    const d = F.dmpnnGraph(P); const fields = [];
    if (!eq(d.atom, r.d.atom, 1e-7)) fields.push('atom'); if (!eq(d.bond, r.d.bond)) fields.push('bond');
    if (!eq(d.src, r.d.src) || !eq(d.dst, r.d.dst) || !eq(d.rev, r.d.rev)) fields.push('index');
    if (fields.length) bad.push(['dmpnn', r.smiles, fields.join(',')]); else okD++;
  } catch (e) { bad.push(['dmpnn-err', r.smiles, e.message]); }
  let g = null; try { g = F.ghgnnGraph(P); } catch (e) { g = null; }
  if (!r.g) { gUnsupPy++; if (g) bad.push(['ghgnn-js-accepts-unsupported', r.smiles]); continue; }
  nG++; if (!g) { gUnsupJS++; bad.push(['ghgnn-js-rejects', r.smiles]); continue; }
  const f = [];
  if (!eq(g.x, r.g.x)) f.push('x'); if (!eq(g.ea, r.g.ea)) f.push('ea'); if (!eq([g.row, g.col], r.g.ei)) f.push('ei');
  for (const k of ['ap', 'bp', 'tpsa']) if (Math.abs(g[k] - r.g[k]) > 1e-9) f.push(k);
  for (const k of ['hba', 'hbd']) if (g[k] !== r.g[k]) f.push(k);
  if (f.length) bad.push(['ghgnn', r.smiles, f.join(',')]); else okG++;
}
console.log(JSON.stringify({ molecules: n, dmpnn_identical: okD, ghgnn_checked: nG, ghgnn_identical: okG, ghgnn_unsupported_python: gUnsupPy, mismatches: bad.length }));
const kinds = {}; for (const b of bad) kinds[b[0] + ':' + (b[2] || '')] = (kinds[b[0] + ':' + (b[2] || '')] || 0) + 1; console.log(kinds);
fs.writeFileSync('featurize_mismatches.json', JSON.stringify(bad, null, 1)); console.log(bad.slice(0, 15));
