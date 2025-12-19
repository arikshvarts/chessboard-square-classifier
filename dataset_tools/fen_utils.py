from typing import List

PIECE_TO_ID = {
    "empty": 0,
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}

ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}


def fen_board_to_64_labels(fen: str) -> List[int]:
    """
    Convert a FEN string into 64 label IDs in row-major order.
    row 0 = rank 8 (top), col 0 = file a (left).
    """
    board_part = fen.split(" ")[0]
    ranks = board_part.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Bad FEN ranks: {fen}")

    labels: List[int] = []
    for rank in ranks:  # rank8 -> rank1
        for ch in rank:
            if ch.isdigit():
                labels.extend([PIECE_TO_ID["empty"]] * int(ch))
            else:
                if ch not in PIECE_TO_ID:
                    raise ValueError(f"Unknown FEN piece char '{ch}' in {fen}")
                labels.append(PIECE_TO_ID[ch])

    if len(labels) != 64:
        raise ValueError(
            f"FEN did not expand to 64 squares (got {len(labels)}): {fen}"
        )
    return labels


def idx_to_square_name(square_idx: int) -> str:
    row = square_idx // 8  # 0..7 top->bottom
    col = square_idx % 8   # 0..7 left->right
    file_char = chr(ord("a") + col)
    rank_num = 8 - row
    return f"{file_char}{rank_num}"
