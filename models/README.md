# Production models

The models used by the web tool, trained on all 68,763 measurements with three seeds each (`training/README.md`, section 4). The web tool averages the three seeds of each model.

| Folder | Contents |
|---|---|
| `dmpnn/seed_*/` | D-MPNN trained on activity coefficients and selectivities (λ = 1) for 44 epochs: parameters (`final_params.msgpack`), configuration, scaling of the targets, training history and run record |
| `ghgnn/seed_*/` | GH-GNN trained for 26 epochs (`final.pt`), training history and run record. GH-GNN cannot represent the four ionic liquids with Al, Co, Fe or Sb, so it was trained on 68,538 measurements |
| `acceptance/` | Checks of the production models: fit to the training data, equality of predictions from SMILES and from the training code, and equality on CPU and GPU on a fixed grid of 5 solvents × all solutes × 3 temperatures (`grid_*_cpu.csv`, `grid_*_gpu.csv`) |

Predictions from SMILES: `training/dmpnn/shared/predict_dmpnn.py` and `training/ghgnn/predict_ghgnn.py`.
