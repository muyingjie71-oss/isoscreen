# Training

Code for the three models of the paper and for the production models of the web tool.

| Folder | Contents |
|---|---|
| `dmpnn/` | Directed message-passing neural network (D-MPNN) in JAX/Flax. `configs/dmpnn.yaml` trains on activity coefficients and on the selectivities of isomer pairs (λ = 1), `configs/dmpnn_activities_only.yaml` on activity coefficients only (λ = 0) |
| `ghgnn/` | Gibbs–Helmholtz graph neural network (GH-GNN) of Sanchez Medina and Minor (MIT licence, `ghgnn/ghgnn/LICENSE_GHGNN_MIT.txt`), retrained from scratch with the published hyperparameters |
| `inputs/` | Files the training code needs besides the public dataset: the molecular structures as used for training, their graph cache and the checksums of the source tables |
| `prepare_data.py` | Builds the training inputs from `data/` |

ECFP + LightGBM is in `validation/ecfp_lightgbm/`.

## 1. Prepare the inputs

```bash
python training/prepare_data.py
```

This writes `training/data/new_isomer_test/` and `training/data/new_solvent_test/` and checks that they equal the training data of the paper. The structures are those used for training, before the eight corrections listed in `data/structure_corrections.csv`.

## 2. New-isomer test (whole isomer families held out)

Each job trains one model on one fold. The five folds of three seeds give the out-of-fold predictions.

```bash
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
D=training/data/new_isomer_test
for cfg in dmpnn dmpnn_activities_only; do
  for seed in 20261007 20261008 20261009; do
    for fold in 1 2 3 4 5; do
      python training/dmpnn/shared/run_matrix_job.py --data $D --config training/dmpnn/configs/$cfg.yaml \
        --seed $seed --fold $fold --out runs/new_isomer/$(grep comparison training/dmpnn/configs/$cfg.yaml | cut -d' ' -f2)_seed$seed
    done
  done
done
python training/dmpnn/shared/aggregate.py --data $D --root runs/new_isomer \
  --configs training/dmpnn/configs/dmpnn.yaml training/dmpnn/configs/dmpnn_activities_only.yaml --seeds 20261007 20261008 20261009

for seed in 20261007 20261008 20261009; do
  for fold in 1 2 3 4 5; do
    python training/ghgnn/run_ghgnn_job.py --data $D --seed $seed --fold $fold --out runs/new_isomer/GHGNN_seed$seed
  done
done
python training/ghgnn/aggregate_ghgnn.py --data $D --root runs/new_isomer --seeds 20261007 20261008 20261009
```

## 3. New-solvent test (whole solvents held out)

The same commands with `D=training/data/new_solvent_test`, the D-MPNN configuration `dmpnn.yaml` only, and `runs/new_solvent` as output folder.

## 4. Production models for the web tool

All 68,763 measurements are used for training, for a fixed number of epochs (the median best epoch of the new-solvent test).

```bash
D=training/data/new_isomer_test
for seed in 20261007 20261008 20261009; do
  python training/dmpnn/shared/run_production_job.py --data $D --config training/dmpnn/configs/dmpnn.yaml \
    --seed $seed --fixed_epochs 44 --out runs/production/dmpnn/seed_$seed
  python training/ghgnn/run_ghgnn_job.py --data $D --seed $seed --fold 0 --fixed_epochs 26 --out runs/production/ghgnn/seed_$seed
done
python training/dmpnn/shared/production_check.py --data $D --models runs/production/dmpnn --report dmpnn_check.json --grid dmpnn_grid.csv
python training/ghgnn/production_check_ghgnn.py --data $D --models runs/production/ghgnn --report ghgnn_check.json --grid ghgnn_grid.csv
```

The models used by the web tool are in `models/`, with the results of these checks in `models/acceptance/`. GH-GNN writes its model into a subfolder `Fold_0/`.

## Environment

D-MPNN: Python 3.12 with the packages in `dmpnn/shared/requirements.txt`. GH-GNN: PyTorch 2, PyTorch Geometric and RDKit. All runs of the paper used one NVIDIA A100 GPU. D-MPNN runs are deterministic on GPU with the `XLA_FLAGS` setting above, so a run with the same seed gives the same parameters.

The records of the runs in the paper are in `validation/runs/`.
