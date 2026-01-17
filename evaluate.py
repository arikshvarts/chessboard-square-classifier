"""
OFFICIAL EVALUATION API - DO NOT MODIFY SIGNATURE
This file provides the exact predict_board() function required by the course spec.

Required by: Intro to Deep Learning (Fall 2025) - Final Project
Projects 1 & 2: Chessboard State Prediction

Function Signature (EXACT):
    def predict_board(image: np.ndarray) -> torch.Tensor

Input Specification:
    - image: numpy.ndarray
    - Shape: (H, W, 3)
    - Channel order: RGB
    - Dtype: uint8
    - Value range: [0, 255]

Output Specification:
    - Type: torch.Tensor
    - Shape: (8, 8)
    - Device: CPU
    - Dtype: torch.int64

Class Encoding (EXACT - DO NOT CHANGE):
    0: White Pawn
    1: White Rook
    2: White Knight
    3: White Bishop
    4: White Queen
    5: White King
    6: Black Pawn
    7: Black Rook
    8: Black Knight
    9: Black Bishop
    10: Black Queen
    11: Black King
    12: Empty Square
    13: Out-of-Distribution (OOD) / Unknown / Invalid

Board Coordinate Convention:
    output[0, 0] = top-left square of the image
    output[0, 7] = top-right square
    output[7, 0] = bottom-left square
    output[7, 7] = bottom-right square
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_tools'))

from model import ChessSquareClassifier
from extract_squares import extract_64_square_crops

# ============================================================================
# CLASS MAPPING: OFFICIAL SPEC -> INTERNAL MODEL
# ============================================================================

# Official spec encoding (what we must return)
SPEC_ENCODING = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11,  # Black pieces
    'empty': 12,
    'OOD': 13
}

# Internal model encoding (what our trained model outputs)
# This matches dataset_tools/fen_utils.py PIECE_TO_ID
INTERNAL_ENCODING = {
    'empty': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,  # Black pieces
}

# Create mapping: internal model output -> spec encoding
INTERNAL_TO_SPEC = {
    0: 12,  # empty -> 12
    1: 0,   # P -> 0
    2: 2,   # N -> 2
    3: 3,   # B -> 3
    4: 1,   # R -> 1
    5: 4,   # Q -> 4
    6: 5,   # K -> 5
    7: 6,   # p -> 6
    8: 8,   # n -> 8
    9: 9,   # b -> 9
    10: 7,  # r -> 7
    11: 10, # q -> 10
    12: 11, # k -> 11
}

# Reverse mapping for reference
SPEC_TO_INTERNAL = {v: k for k, v in INTERNAL_TO_SPEC.items()}

# ============================================================================
# GLOBAL MODEL STATE
# ============================================================================

_MODEL = None
_DEVICE = None
_TRANSFORM = None
_MODEL_LOADED = False


def _initialize_model():
    """Initialize model once (lazy loading)"""
    global _MODEL, _DEVICE, _TRANSFORM, _MODEL_LOADED
    
    if _MODEL_LOADED:
        return
    
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transform
    _TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Try to load checkpoint
    checkpoint_paths = [
        'checkpoints/best_model.pth',
        'checkpoints/fold_1/best_model.pth',
        'dataset_out/best_model_fold_1.pth',
    ]
    
    checkpoint_loaded = False
    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            try:
                print(f"Loading checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
                
                # Check if checkpoint is from Colab training or local training
                state_dict = checkpoint['model_state_dict']
                if 'backbone.conv1.weight' not in state_dict and 'conv1.weight' in state_dict:
                    # Colab checkpoint - load directly into ResNet50
                    print("  Format: Colab (direct ResNet50)")
                    from torchvision import models
                    _MODEL = models.resnet50(weights=None)
                    _MODEL.fc = nn.Linear(_MODEL.fc.in_features, 13)
                    _MODEL.load_state_dict(state_dict)
                    _MODEL.to(_DEVICE)
                    _MODEL.eval()
                else:
                    # Local checkpoint - use ChessSquareClassifier wrapper
                    print("  Format: Local (ChessSquareClassifier)")
                    _MODEL = ChessSquareClassifier(num_classes=13, pretrained=False, model_name='resnet50')
                    _MODEL.load_state_dict(state_dict)
                    _MODEL.to(_DEVICE)
                    _MODEL.eval()
                
                checkpoint_loaded = True
                print(f"✓ Model loaded from {ckpt_path}")
                if 'val_acc' in checkpoint:
                    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
                break
            except Exception as e:
                print(f"⚠ Failed to load {ckpt_path}: {e}")
                continue
    
    if not checkpoint_loaded:
        print("⚠ No trained checkpoint found. Using pretrained ResNet50 (not chess-specific).")
        print("  For accurate predictions, place trained model at: checkpoints/best_model.pth")
        _MODEL = ChessSquareClassifier(num_classes=13, pretrained=True, model_name='resnet50')
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    
    _MODEL_LOADED = True


# ============================================================================
# OFFICIAL EVALUATION FUNCTION
# ============================================================================

def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    OFFICIAL EVALUATION FUNCTION - Predict chessboard state from RGB image.
    
    This function signature MUST NOT be changed - it's specified by the course requirements.
    
    Args:
        image: numpy.ndarray of shape (H, W, 3), RGB, uint8, range [0, 255]
        
    Returns:
        torch.Tensor of shape (8, 8), dtype torch.int64, on CPU
        Values in range [0, 13] following the official class encoding:
            0-5: White pieces (P, R, N, B, Q, K)
            6-11: Black pieces (p, r, n, b, q, k)
            12: Empty square
            13: OOD/Unknown/Invalid
            
    Coordinate convention:
        output[0, 0] = top-left square of input image
        output[7, 7] = bottom-right square of input image
    """
    # Lazy load model on first call
    if not _MODEL_LOADED:
        _initialize_model()
    
    try:
        # Convert numpy array to PIL Image
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be np.ndarray, got {type(image)}")
        
        if image.dtype != np.uint8:
            raise ValueError(f"Input dtype must be uint8, got {image.dtype}")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Input shape must be (H, W, 3), got {image.shape}")
        
        pil_image = Image.fromarray(image, mode='RGB')
        
        # Extract 64 square crops
        squares = extract_64_square_crops(pil_image)
        
        if squares is None or len(squares) != 64:
            # Board detection failed - return all OOD (13)
            print("⚠ Board detection failed, returning all OOD")
            return torch.full((8, 8), 13, dtype=torch.int64, device='cpu')
        
        # Initialize output tensor
        board_tensor = torch.zeros((8, 8), dtype=torch.int64, device='cpu')
        
        # Predict each square
        with torch.no_grad():
            for square_idx, square_img in enumerate(squares):
                # Transform and predict
                img_tensor = _TRANSFORM(square_img).unsqueeze(0).to(_DEVICE)
                output = _MODEL(img_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = probs.max(1)
                
                internal_class = predicted.item()
                confidence_val = confidence.item()
                
                # Convert internal encoding to spec encoding
                if internal_class in INTERNAL_TO_SPEC:
                    spec_class = INTERNAL_TO_SPEC[internal_class]
                else:
                    # Unknown class from model -> OOD
                    spec_class = 13
                
                # Note: Low confidence threshold disabled by default
                # The spec doesn't require confidence filtering - model should be confident
                # Uncomment below if you want to mark low-confidence predictions as OOD:
                # if confidence_val < 0.3:
                #     spec_class = 13
                
                # Place in board tensor
                row = square_idx // 8
                col = square_idx % 8
                board_tensor[row, col] = spec_class
        
        return board_tensor
        
    except Exception as e:
        # On any error, return all OOD
        print(f"⚠ Error in predict_board: {e}")
        return torch.full((8, 8), 13, dtype=torch.int64, device='cpu')


# ============================================================================
# HELPER FUNCTIONS FOR DEBUGGING
# ============================================================================

def spec_tensor_to_fen(board_tensor: torch.Tensor) -> str:
    """
    Convert spec-encoded board tensor to FEN notation for visualization.
    
    Args:
        board_tensor: (8, 8) tensor with spec encoding (0-13)
        
    Returns:
        FEN string (board part only, no move counters)
    """
    SPEC_TO_PIECE = {
        0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
        6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
        12: '', 13: '?'
    }
    
    fen_ranks = []
    for row in range(8):
        fen_rank = ""
        empty_count = 0
        
        for col in range(8):
            piece_id = board_tensor[row, col].item()
            piece = SPEC_TO_PIECE.get(piece_id, '?')
            
            if piece == '':  # Empty
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece
        
        if empty_count > 0:
            fen_rank += str(empty_count)
        
        fen_ranks.append(fen_rank)
    
    return '/'.join(fen_ranks)


def print_board(board_tensor: torch.Tensor):
    """Print board in human-readable format"""
    SPEC_TO_PIECE = {
        0: '♙', 1: '♖', 2: '♘', 3: '♗', 4: '♕', 5: '♔',
        6: '♟', 7: '♜', 8: '♞', 9: '♝', 10: '♛', 11: '♚',
        12: '·', 13: '?'
    }
    
    print("\n  a b c d e f g h")
    for row in range(8):
        print(f"{8-row} ", end="")
        for col in range(8):
            piece_id = board_tensor[row, col].item()
            piece = SPEC_TO_PIECE.get(piece_id, '?')
            print(f"{piece} ", end="")
        print(f"{8-row}")
    print("  a b c d e f g h\n")


def save_board_visualization(image: np.ndarray, board_tensor: torch.Tensor, output_path: str = './results/prediction.png'):
    """
    Save visualization of predicted board with OOD squares marked with red X.
    Required by instructor for Project 1.
    
    Args:
        image: Original input image (H, W, 3) RGB numpy array
        board_tensor: Predicted board tensor (8, 8)
        output_path: Where to save the output image
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else './results', exist_ok=True)
    
    # Convert to PIL image
    pil_img = Image.fromarray(image, mode='RGB')
    draw = ImageDraw.Draw(pil_img)
    
    # Get image dimensions
    w, h = pil_img.size
    sq_w = w // 8
    sq_h = h // 8
    
    # Draw red X on OOD squares (value 13)
    for row in range(8):
        for col in range(8):
            if board_tensor[row, col].item() == 13:  # OOD square
                # Calculate square boundaries
                left = col * sq_w
                upper = row * sq_h
                right = (col + 1) * sq_w
                lower = (row + 1) * sq_h
                
                # Draw red X
                line_width = max(3, min(sq_w, sq_h) // 20)
                draw.line([(left, upper), (right, lower)], fill='red', width=line_width)
                draw.line([(right, upper), (left, lower)], fill='red', width=line_width)
    
    # Save
    pil_img.save(output_path)
    print(f"Saved visualization to: {output_path}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test official predict_board() function')
    parser.add_argument('--image', type=str, required=True, help='Path to chess board image')
    parser.add_argument('--show-tensor', action='store_true', help='Show raw tensor values')
    parser.add_argument('--save-viz', action='store_true', help='Save visualization with OOD marked (Project 1 requirement)')
    parser.add_argument('--output', type=str, default='./results/prediction.png', help='Output path for visualization')
    args = parser.parse_args()
    
    # Load image as numpy array
    pil_img = Image.open(args.image).convert('RGB')
    image_np = np.array(pil_img)
    
    print("="*60)
    print("OFFICIAL EVALUATION API TEST")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Shape: {image_np.shape}")
    print(f"Dtype: {image_np.dtype}")
    print(f"Range: [{image_np.min()}, {image_np.max()}]")
    
    # Call official function
    result = predict_board(image_np)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Output type: {type(result)}")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Output device: {result.device}")
    print(f"Value range: [{result.min().item()}, {result.max().item()}]")
    
    if args.show_tensor:
        print("\nRaw tensor:")
        print(result)
    
    print("\nVisualization:")
    print_board(result)
    
    print("FEN notation:")
    fen = spec_tensor_to_fen(result)
    print(f"  {fen}")
    
    # Save visualization if requested
    if args.save_viz:
        print("\n" + "="*60)
        print("SAVING VISUALIZATION (Project 1 Requirement)")
        print("="*60)
        save_board_visualization(image_np, result, args.output)
    
    print("\n" + "="*60)
    print("✓ Test complete - function signature is compliant")
    print("="*60)
