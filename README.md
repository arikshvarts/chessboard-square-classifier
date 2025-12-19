# Chessboard Square Classifier

Pipeline to convert labeled chess video frames into per-square training data and evaluate predictions.

**Team**: Ariel Shvarts, Nikol Koifman

**Context**: Course project (Intro to Deep Learning, Fall 2025) delivering a dataset factory, evaluation harness, and debug visuals for chessboard square classification.

![Sample debug grid](docs/assets/sample_debug_grid.png)

## Repo layout
- `Data/` raw per-game folders (`gameX_per_frame/` with `tagged_images/` and `gameX.csv`).
- `dataset_tools/` dataset factory, FEN utilities, debug visualizations, evaluation.
- `dataset_out/` generated artifacts (manifest, classes.json, debug images) — typically not committed.

## Setup
1. Create a virtual environment (example): `python -m venv .venv`
2. Activate (PowerShell): `.\.venv\Scripts\Activate.ps1`
3. Install deps: `pip install -r requirements.txt`

## Dataset build (Phase 1)
- Default manifest + class map (80/20 row-wise, seed=42): `python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out`
- Adjust splits/seed: `--train_ratio 0.75 --val_ratio 0.05 --seed 1234`
- Optional board detect + warp before the 8x8 grid: add `--detect_board --warp_size 800` (requires `opencv-python`).
- Optional: generate debug grids to verify orientation.

Board detect helper (standalone): `python -m dataset_tools.board_detect_and_warp --image <frame> --out_warp <out.png> --out_debug <debug.png>`.

## Additional datasets (optional)
- Lichess PGN/position dumps: https://database.lichess.org/
- Kaggle Chess Piece Images dataset: [https://www.kaggle.com/datasets/koryakinp/chess-pieces-images](https://www.kaggle.com/datasets/koryakinp/chess-positions)
- https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2

## Evaluation (later)
- Compare predictions vs manifest: `python dataset_tools/eval.py --manifest dataset_out/dataset_manifest.csv --preds path/to/preds.csv`

## Git workflow
- Start: `git checkout -b feature/<task>`
- Sync with base: `git checkout main` → `git pull` → `git checkout feature/<task>` → `git merge main`
- Commit/push: `git add ...` → `git commit -m "..."` → `git push -u origin feature/<task>`
- Open PR: base = `main`, compare = `feature/<task>`; request review; merge when green.
- After merge: `git checkout main` → `git pull` → `git branch -d feature/<task>` → `git push origin --delete feature/<task>`
- Keep data out of Git: `Data/`, `dataset_out/`, `checkpoints/`, `outputs/` stay in `.gitignore`.


