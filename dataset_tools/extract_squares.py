from typing import List
from PIL import Image


def extract_64_square_crops(img: Image.Image) -> List[Image.Image]:
    """
    Assumes the image is already a tight crop of the board.
    Returns 64 PIL crops in square_idx order (row-major).
    """
    w, h = img.size
    sq_w = w // 8
    sq_h = h // 8

    crops: List[Image.Image] = []
    for row in range(8):
        for col in range(8):
            left = col * sq_w
            upper = row * sq_h
            right = (col + 1) * sq_w
            lower = (row + 1) * sq_h
            crops.append(img.crop((left, upper, right, lower)))
    return crops
