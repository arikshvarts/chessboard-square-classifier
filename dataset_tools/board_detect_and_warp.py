"""
Optional board detection + perspective warp helper.
Not used unless called explicitly. Requires OpenCV (`pip install opencv-python`).
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_board_quad(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return 4-point contour of the largest roughly-quadrilateral board candidate."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:10]:  # examine top contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return None


def warp_board(image_bgr: np.ndarray, quad: np.ndarray, out_size: int = 800) -> np.ndarray:
    """Perspective-warp the image to a square of size out_size x out_size."""
    rect = order_points(quad)
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_size, out_size))
    return warped


def save_debug(image_bgr: np.ndarray, quad: np.ndarray, out_path: str) -> None:
    vis = image_bgr.copy()
    cv2.polylines(vis, [quad.astype(int)], isClosed=True, color=(0, 0, 255), thickness=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def process_image(image_path: str, out_warp: str, out_debug: str | None = None, out_size: int = 800) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    quad = find_board_quad(image)
    if quad is None:
        raise RuntimeError("Board contour not found; try tuning thresholds or pre-cropping")

    warped = warp_board(image, quad, out_size=out_size)
    os.makedirs(os.path.dirname(out_warp), exist_ok=True)
    cv2.imwrite(out_warp, warped)

    if out_debug:
        save_debug(image, quad, out_debug)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect board, warp to square, save warped image")
    ap.add_argument("--image", required=True, help="Input frame image path")
    ap.add_argument("--out_warp", required=True, help="Output path for warped square board")
    ap.add_argument("--out_debug", help="Optional output path for contour overlay debug")
    ap.add_argument("--out_size", type=int, default=800, help="Warp output size (pixels)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    process_image(args.image, args.out_warp, args.out_debug, out_size=args.out_size)


if __name__ == "__main__":
    main()
