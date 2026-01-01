"""
Chess Board Position Predictor
Usage: python src/predict.py --image path/to/board.jpg
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model import ChessSquareClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset_tools.extract_squares import extract_64_square_crops


def load_model(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ChessSquareClassifier(num_classes=13, pretrained=False, model_name='resnet50')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")
    return model, checkpoint


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_square(model, square_img, transform, device, confidence_threshold=0.7):
    img_tensor = transform(square_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = probs.max(1)
    
    confidence = confidence.item()
    predicted = predicted.item()
    is_certain = confidence >= confidence_threshold
    
    return predicted, confidence, is_certain


def predictions_to_fen(predictions, class_map):
    id_to_piece = {int(k): v for k, v in class_map.items()}
    board = [[None for _ in range(8)] for _ in range(8)]
    
    for square_idx in range(64):
        class_id, confidence, is_certain = predictions[square_idx]
        row = square_idx // 8
        col = square_idx % 8
        
        if not is_certain:
            board[row][col] = '?'
        else:
            piece = id_to_piece[class_id]
            if piece != 'empty':
                board[row][col] = piece
    
    fen_ranks = []
    for rank in range(8):
        fen_rank = ""
        empty_count = 0
        
        for file in range(8):
            piece = board[rank][file]
            
            if piece is None:
                empty_count += 1
            elif piece == '?':
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += '?'
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece
        
        if empty_count > 0:
            fen_rank += str(empty_count)
        
        fen_ranks.append(fen_rank)
    
    return '/'.join(fen_ranks)


def predict_board(image_path, model, classes_json, device='cpu', 
                  confidence_threshold=0.7, verbose=True):
    if not os.path.exists(classes_json):
        raise FileNotFoundError(f"Classes file not found: {classes_json}")
    
    with open(classes_json, 'r') as f:
        class_map = json.load(f)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    if verbose:
        print(f"Image size: {img.size}")
    
    all_squares = extract_64_square_crops(img)
    if verbose:
        print(f"Extracted 64 squares")
    
    transform = get_transform()
    predictions = []
    
    for square_idx, square_img in enumerate(all_squares):
        pred = predict_square(model, square_img, transform, device, confidence_threshold)
        predictions.append(pred)
        
        if verbose:
            class_id, conf, certain = pred
            symbol = class_map[str(class_id)]
            status = "OK" if certain else "?"
            print(f"  Square {square_idx:2d}: {symbol:5s} ({conf:6.1%}) [{status}]")
    
    fen = predictions_to_fen(predictions, class_map)
    return fen, predictions


def visualize_fen(fen, output_path='predicted_board.svg'):
    try:
        import chess
        import chess.svg
        
        clean_fen = fen.replace('?', '1')
        full_fen = clean_fen + " w KQkq - 0 1"
        
        board = chess.Board(full_fen)
        svg_data = chess.svg.board(board, size=400)
        
        with open(output_path, 'w') as f:
            f.write(svg_data)
        
        print(f"Saved board diagram to: {output_path}")
        
    except ImportError:
        print("python-chess not installed. Install with: pip install python-chess")
    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description='Predict chess board position from image')
    parser.add_argument('--image', type=str, required=True, help='Path to chessboard image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, default='dataset_out/classes.json', help='Path to classes JSON')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for occlusion detection')
    parser.add_argument('--output', type=str, default='predicted.fen', help='Output FEN file')
    parser.add_argument('--visualize', action='store_true', help='Generate chess board visualization')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, device)
    
    print(f"\nPredicting: {args.image}")
    print("=" * 60)
    
    fen, predictions = predict_board(
        args.image,
        model,
        args.classes,
        device,
        args.confidence_threshold,
        verbose=not args.quiet
    )
    
    print("=" * 60)
    print(f"\nFEN: {fen}\n")
    
    with open(args.output, 'w') as f:
        f.write(fen)
    print(f"Saved FEN to: {args.output}")
    
    if args.visualize:
        visualize_fen(fen)


if __name__ == '__main__':
    main()