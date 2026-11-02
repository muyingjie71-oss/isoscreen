# Web tool

IsoScreen runs entirely in the browser. It converts the two isomers to molecular graphs with RDKit compiled to WebAssembly, encodes them with the solute part of both models, and evaluates the solute–solvent part of each model against 6,911 solvents encoded in advance, using ONNX Runtime Web. Predictions are the mean of D-MPNN and GH-GNN, each averaged over three training runs.

| Folder | Contents |
|---|---|
| `site/` | The static site, deployed to GitHub Pages by `.github/workflows/pages.yml` |
| `build/` | Scripts that build `site/data` and `site/model` from the production models in `models/` |
| `solvent_library/` | The 6,911 solvents, the ions they are built from and the list of Langner and Brabec |
| `tests/` | Checks that the browser reproduces the Python models |

## Running the site locally

```bash
cd webtool/site
python3 -m http.server 8000
# open http://localhost:8000
```

The page must be served over http. Opening `index.html` directly from disk does not work.

## Building the site files

Run from the repository root, after `python training/prepare_data.py`:

```bash
python webtool/build/build_solvent_library.py      # solvent_library/
python webtool/build/export_dmpnn_params.py        # D-MPNN parameters as arrays (needs JAX/Flax)
python webtool/build/verify_split.py               # solute and solvent parts equal the production models
python webtool/build/precompute_solvents.py        # encodings of all solvents
python webtool/build/export_onnx.py                # solute–solvent parts as ONNX
python webtool/build/build_site_data.py            # site/data and site/model
```

Intermediate files go to `build/work/`. The split models agree with the production models to within 9 × 10⁻⁶ in ln γ∞. The site stores the ONNX models and the encoders in single precision and the solvent encodings and the GH-GNN edge network in half precision.

## Tests

```bash
cd webtool/tests
npm install @rdkit/rdkit onnxruntime-web
python make_reference.py ref_features.jsonl && node featurize_test.mjs     # features equal the Python implementation
node screen_test.mjs                                                       # predictions on the check grid, and timing
node isotype_test.mjs                                                      # isomer types against the dataset labels
```

`screen_test.mjs` compares the browser pipeline with the Python models on the 5,040 points of `models/acceptance/grid_*_cpu.csv`. The largest difference in ln γ∞ is 2 × 10⁻⁴. The interface tests (`ui_*.py`, Playwright) expect the site served at the address in `ISOSCREEN_URL`.

## Solvent library

The library holds the 767 measured solvents that both models can represent, 5,759 ionic liquids combined from the measured cations and anions, and 385 molecular solvents from the CHEM21 guide and from Langner and Brabec. Each unmeasured solvent is assigned its highest Tanimoto similarity to the measured solvents (Morgan fingerprints, radius 2, 2048 bits), and solvents below 0.3 are treated with the rules for new isomer pairs. Details are in `solvent_library/README_solvent_library.md`.
