import argparse
import glob
import json
import os
from typing import List

import pandas as pd

from dataset_tools.fen_utils import PIECE_TO_ID, fen_board_to_64_labels, idx_to_square_name


def pick_frame_column(columns: List[str]) -> str:
    """Prefer from_frame -> frame_id -> to_frame."""
    candidates = ["from_frame", "frame_id", "to_frame"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"No frame id column found; expected one of {candidates}, got {columns}")


def infer_frame_path(game_dir: str, frame_id: int) -> str | None:
    """Return best-guess frame path for this dataset (frame_000123.jpg/png)."""
    patterns = [
        os.path.join(game_dir, "tagged_images", f"frame_{frame_id:06d}.jpg"),
        os.path.join(game_dir, "tagged_images", f"frame_{frame_id:06d}.png"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def build_manifest(
    data_root: str,
    out_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    skip_missing: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    os.makedirs(out_root, exist_ok=True)

    classes_path = os.path.join(out_root, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in PIECE_TO_ID.items()}, f, indent=2)

    game_dirs = sorted([d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)])
    if not game_dirs:
        raise FileNotFoundError(f"No game directories found under {data_root}")

    rows = []
    for game_dir in game_dirs:
        game_id = os.path.basename(game_dir)

        csv_candidates = glob.glob(os.path.join(game_dir, "*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV found in {game_dir}")
        game_csv = csv_candidates[0]

        df = pd.read_csv(game_csv)
        if "fen" not in df.columns:
            raise ValueError(f"'fen' column not found in {game_csv}. Columns: {list(df.columns)}")

        frame_col = pick_frame_column(list(df.columns))

        for _, r in df.iterrows():
            frame_id = int(r[frame_col])
            fen = str(r["fen"])
            labels64 = fen_board_to_64_labels(fen)

            frame_path = infer_frame_path(game_dir, frame_id)
            if frame_path is None:
                msg = f"Skipping frame_id={frame_id} in {game_id}: file not found"
                if skip_missing:
                    print(msg)
                    continue
                raise FileNotFoundError(msg)

            for square_idx in range(64):
                row = square_idx // 8
                col = square_idx % 8
                rows.append(
                    {
                        "frame_path": frame_path,
                        "game_id": game_id,
                        "frame_id": frame_id,
                        "square_idx": square_idx,
                        "row": row,
                        "col": col,
                        "square_name": idx_to_square_name(square_idx),
                        "label_id": labels64[square_idx],
                    }
                )

    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("test_ratio must be positive; adjust train/val ratios")

    manifest = pd.DataFrame(rows)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(manifest)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    manifest["split"] = "train"
    if n_val > 0:
        manifest.loc[n_train : n_train + n_val - 1, "split"] = "val"
    manifest.loc[n_train + n_val :, "split"] = "test"

    manifest_path = os.path.join(out_root, "dataset_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(
        f"Wrote manifest: {manifest_path} (train={n_train}, val={n_val}, test={n - n_train - n_val}, total={n})"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build dataset manifest from chess frames + FEN labels")
    ap.add_argument("--data_root", default="Data", help="Root folder containing game subfolders")
    ap.add_argument("--out_root", default="dataset_out", help="Output folder for manifest/classes")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (row-wise)")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="Val split ratio (row-wise)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip rows whose frame image is missing (default).",
    )
    ap.add_argument(
        "--no-skip-missing",
        dest="skip_missing",
        action="store_false",
        help="Fail if any frame image is missing.",
    )
    ap.set_defaults(skip_missing=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(
        args.data_root,
        args.out_root,
        args.train_ratio,
        args.val_ratio,
        skip_missing=args.skip_missing,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
