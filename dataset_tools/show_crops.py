import argparse
import os
import random
import pandas as pd

from dataset_tools.debug_grid import save_debug_grid


def sample_frames(df: pd.DataFrame, per_game: int, seed: int):
    random.seed(seed)
    frames = []
    for game_id, gdf in df.groupby("game_id"):
        unique_frames = gdf[["frame_path", "game_id"]].drop_duplicates()
        picks = unique_frames.sample(n=min(per_game, len(unique_frames)), random_state=seed, replace=False)
        frames.extend(picks.to_dict(orient="records"))
    return frames


def build_label_list(df: pd.DataFrame, frame_path: str, game_id: str):
    labels64 = (
        df[(df["frame_path"] == frame_path) & (df["game_id"] == game_id)]
        .sort_values("square_idx")["label_id"]
        .tolist()
    )
    return labels64


def main():
    ap = argparse.ArgumentParser(description="Save debug grids for sampled frames across games")
    ap.add_argument("--manifest", default="dataset_out/dataset_manifest.csv", help="Path to dataset manifest")
    ap.add_argument("--out_dir", default="dataset_out/debug_samples", help="Where to write debug images")
    ap.add_argument("--per_game", type=int, default=2, help="How many frames to sample per game")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.manifest)

    frames = sample_frames(df, per_game=args.per_game, seed=args.seed)
    for rec in frames:
        frame_path = rec["frame_path"]
        game_id = rec["game_id"]
        labels64 = build_label_list(df, frame_path, game_id)
        fname = os.path.splitext(os.path.basename(frame_path))[0]
        out_path = os.path.join(args.out_dir, f"{game_id}_{fname}_debug.png")
        save_debug_grid(frame_path, labels64, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
