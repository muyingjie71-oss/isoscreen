import fs from 'fs'; import { createRequire } from 'module'; const require = createRequire(import.meta.url);
const R = await require('@rdkit/rdkit')(); const { makeIsoType } = await import('../site/js/isotype.js'); const classify = makeIsoType(R);
const sol = Object.fromEntries(JSON.parse(fs.readFileSync('../site/data/solutes.json')).map((s) => [s.id, s.smiles]));
const pairs = JSON.parse(fs.readFileSync('../site/data/pairs.json')); let ok = 0; const bad = [];
for (const p of pairs) { const c = classify(sol[p.a], sol[p.b]); if (c === p.rel) ok++; else bad.push([p.rel, c, sol[p.a], sol[p.b]]); }
console.log(`isomer type, JS heuristic vs dataset labels: ${ok}/${pairs.length}`); console.log(bad);
