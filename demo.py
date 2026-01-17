"""
Demo Script - Quick Test of Chess Board Classifier

This script demonstrates how to use the trained model to predict
a chessboard position from an image file.

Usage:
    python demo.py --image path/to/chessboard.jpg
    python demo.py --image examples/board1.jpg --output demo_results/
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch

from evaluate import predict_board


def visualize_prediction(board_tensor, output_path=None):
    """Print board prediction in readable format."""
    
    piece_names = {
        0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
        6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
        12: '.', 13: '?'
    }
    
    print("\n" + "="*50)
    print("PREDICTED CHESS BOARD")
    print("="*50)
    print("  a b c d e f g h")
    print(" +-----------------+")
    
    for rank in range(8):
        row = []
        for file in range(8):
            piece_id = board_tensor[rank, file].item()
            row.append(piece_names.get(piece_id, '?'))
        print(f"{8-rank}| {' '.join(row)} |{8-rank}")
    
    print(" +-----------------+")
    print("  a b c d e f g h")
    print("="*50)
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        result_file = os.path.join(output_path, "prediction.txt")
        with open(result_file, 'w') as f:
            f.write("Predicted Board:\n")
            for rank in range(8):
                row = [str(board_tensor[rank, file].item()) for file in range(8)]
                f.write(f"Rank {8-rank}: {' '.join(row)}\n")
        print(f"\nResults saved to: {result_file}")


def main():
    parser = argparse.ArgumentParser(description='Demo: Predict chessboard from image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to chessboard image')
    parser.add_argument('--output', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print(f"Please ensure trained model is at: {args.model}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("CHESS BOARD CLASSIFIER - DEMO")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    
    try:
        image = np.array(Image.open(args.image).convert('RGB'))
        print(f"Loaded image: {image.shape}")
        
        print("\nRunning prediction...")
        board_tensor = predict_board(image)
        
        print(f"Prediction complete!")
        print(f"Output shape: {board_tensor.shape}")
        print(f"Output dtype: {board_tensor.dtype}")
        print(f"Output device: {board_tensor.device}")
        
        visualize_prediction(board_tensor, args.output)
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
