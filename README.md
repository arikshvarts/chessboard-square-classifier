# Chessboard Square Classifier

Pipeline to convert labeled chess video frames into per-square training data and evaluate predictions.

## Repo layout
- `Data/` raw per-game folders (`gameX_per_frame/` with `tagged_images/` and `gameX.csv`).
- `dataset_tools/` dataset factory, FEN utilities, debug visualizations, evaluation.
- `dataset_out/` generated artifacts (manifest, classes.json, debug images) — typically not committed.

## Setup
1. Create a virtual environment (example): `python -m venv .venv`
2. Activate (PowerShell): `.\.venv\Scripts\Activate.ps1`
3. Install deps: `pip install -r requirements.txt`

## Dataset build (Phase 1)
Once `dataset_tools/make_dataset.py` is added:
- Generate manifest and class map (default 80/20 train/test row-wise, deterministic seed=42): `python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out`
- Customize split/seed if needed, e.g. `--train_ratio 0.75 --val_ratio 0.05 --seed 1234`
- Optional: generate debug grids to verify orientation.

## Evaluation (later)
- Compare predictions vs manifest: `python dataset_tools/eval.py --manifest dataset_out/dataset_manifest.csv --preds path/to/preds.csv`

## Git workflow (suggested)
- Create feature branches (`feature/dataset-pipeline`, `feature/model-training`).
- Keep `master` runnable; use small, descriptive commits.
- Do not commit large data (`Data/`, `dataset_out/`, `checkpoints/`, `outputs/`).

## Next steps
- Add dataset tools (`fen_utils.py`, `extract_squares.py`, `make_dataset.py`, `debug_grid.py`, `eval.py`).
- Run dataset build and verify with debug grids.
- Hand off manifest to model-training teammate.
