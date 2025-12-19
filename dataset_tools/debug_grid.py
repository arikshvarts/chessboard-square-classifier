import os
from typing import List
from PIL import Image, ImageDraw, ImageFont

from dataset_tools.fen_utils import ID_TO_PIECE, idx_to_square_name
from dataset_tools.extract_squares import extract_64_square_crops


def save_debug_grid(frame_path: str, labels64: List[int], out_path: str) -> None:
    """
    Save a debug grid showing crops with square names and labels.
    """
    img = Image.open(frame_path).convert("RGB")
    crops = extract_64_square_crops(img)

    tile_w, tile_h = crops[0].size
    grid = Image.new("RGB", (tile_w * 8, tile_h * 8), (0, 0, 0))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, tile_h // 6))
    except Exception:
        font = ImageFont.load_default()

    for idx, crop in enumerate(crops):
        row, col = divmod(idx, 8)
        x, y = col * tile_w, row * tile_h
        grid.paste(crop, (x, y))

        piece = ID_TO_PIECE[labels64[idx]]
        sq = idx_to_square_name(idx)
        text = f"{sq}:{piece}"
        draw.rectangle([x, y, x + tile_w, y + int(tile_h * 0.18)], fill=(0, 0, 0))
        draw.text((x + 3, y + 1), text, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
