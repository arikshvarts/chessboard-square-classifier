import argparse
import json
import pandas as pd


def evaluate(manifest_csv: str, preds_csv: str):
    """Compute per-square, piece-only, and per-board accuracy."""
    gt = pd.read_csv(manifest_csv)
    pr = pd.read_csv(preds_csv)

    m = gt.merge(pr, on=["frame_path", "square_idx"], how="inner")

    per_square_acc = (m["label_id"] == m["pred_label_id"]).mean()

    piece_mask = m["label_id"] != 0
    piece_acc = (
        (m.loc[piece_mask, "label_id"] == m.loc[piece_mask, "pred_label_id"]).mean()
        if piece_mask.any()
        else float("nan")
    )

    per_frame = m.groupby("frame_path").apply(lambda df: (df["label_id"] == df["pred_label_id"]).all())
    board_acc = per_frame.mean()

    return {
        "per_square_acc": float(per_square_acc),
        "piece_only_acc": float(piece_acc),
        "board_acc": float(board_acc),
        "n_frames_eval": int(per_frame.shape[0]),
        "n_squares_eval": int(m.shape[0]),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Ground-truth manifest CSV")
    ap.add_argument("--preds", required=True, help="Predictions CSV with frame_path, square_idx, pred_label_id")
    return ap.parse_args()


def main():
    args = parse_args()
    metrics = evaluate(args.manifest, args.preds)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
