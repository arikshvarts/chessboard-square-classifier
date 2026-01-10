"""
Interactive Chess Board Classifier Web App
A beautiful web interface for chess position recognition
"""

import os
import sys
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_tools'))

from model import ChessSquareClassifier
from extract_squares import extract_64_square_crops

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model and settings
MODEL = None
DEVICE = None
CLASS_MAP = None
TRANSFORM = None


def load_class_map():
    """Load the class mapping from classes.json (internal encoding)"""
    classes_path = os.path.join('dataset_out', 'classes.json')
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            return json.load(f)
    else:
        # Default class map matching fen_utils.py (INTERNAL encoding)
        # 0=empty, 1-6=white pieces, 7-12=black pieces
        return {
            "0": "empty", "1": "P", "2": "N", "3": "B", "4": "R", 
            "5": "Q", "6": "K", "7": "p", "8": "n", "9": "b", 
            "10": "r", "11": "q", "12": "k"
        }


def initialize_model():
    """Initialize the model for predictions (uses internal encoding)"""
    global MODEL, DEVICE, CLASS_MAP, TRANSFORM
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    CLASS_MAP = load_class_map()
    
    # Define transform
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Try to load model checkpoint
    checkpoint_dir = 'checkpoints'
    checkpoint_path = None
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            checkpoints.sort()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        
        # Check if checkpoint is from Colab training (direct ResNet50) or local training (ChessSquareClassifier wrapper)
        state_dict = checkpoint['model_state_dict']
        if 'backbone.conv1.weight' not in state_dict and 'conv1.weight' in state_dict:
            # Colab checkpoint - load directly into ResNet50
            print("Detected Colab checkpoint format (direct ResNet50)")
            from torchvision import models
            MODEL = models.resnet50(weights=None)
            MODEL.fc = torch.nn.Linear(MODEL.fc.in_features, 13)
            MODEL.load_state_dict(state_dict)
            MODEL.to(DEVICE)
            MODEL.eval()
        else:
            # Local checkpoint - use ChessSquareClassifier wrapper
            print("Detected local checkpoint format (ChessSquareClassifier)")
            MODEL = ChessSquareClassifier(num_classes=13, pretrained=False, model_name='resnet50')
            MODEL.load_state_dict(state_dict)
            MODEL.to(DEVICE)
            MODEL.eval()
        
        print(f"Model loaded successfully! Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("No checkpoint found. Model will be initialized without pre-trained weights.")
        print("The app will run in demo mode - predictions may not be accurate.")
        MODEL = ChessSquareClassifier(num_classes=13, pretrained=True, model_name='resnet50')
        MODEL.to(DEVICE)
        MODEL.eval()


def predict_board(image):
    """
    Predict chess board position from image
    Returns: dict with board state and visualization data
    """
    try:
        # Extract 64 squares from the board
        squares = extract_64_square_crops(image)
        
        if squares is None or len(squares) != 64:
            return {"error": "Could not detect chess board. Please upload a clear image of a chess board."}
        
        # Predict each square
        predictions = []
        board_state = [[None for _ in range(8)] for _ in range(8)]
        
        id_to_piece = {int(k): v for k, v in CLASS_MAP.items()}
        
        for square_idx, square_img in enumerate(squares):
            img_tensor = TRANSFORM(square_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = MODEL(img_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = probs.max(1)
            
            confidence = confidence.item()
            predicted_class = predicted.item()
            piece = id_to_piece[predicted_class]
            
            row = square_idx // 8
            col = square_idx % 8
            
            board_state[row][col] = {
                'piece': piece,
                'confidence': confidence,
                'square': f"{chr(97 + col)}{8 - row}"  # e.g., 'e4'
            }
            
            predictions.append({
                'square_idx': square_idx,
                'piece': piece,
                'confidence': confidence
            })
        
        # Generate FEN notation
        fen = board_to_fen(board_state)
        
        return {
            'success': True,
            'board_state': board_state,
            'fen': fen,
            'predictions': predictions
        }
        
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}


def board_to_fen(board_state):
    """Convert board state to FEN notation"""
    fen_ranks = []
    
    # Piece mapping - handles both formats
    piece_map = {
        # Standard FEN notation (already correct)
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'p', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q', 'k': 'k',
        # Legacy format (if present)
        'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B',
        'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
        'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b',
        'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k',
        'empty': ''
    }
    
    for rank in range(8):
        fen_rank = ""
        empty_count = 0
        
        for file in range(8):
            piece_info = board_state[rank][file]
            piece = piece_info['piece']
            
            if piece == 'empty':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece_map.get(piece, '?')
        
        if empty_count > 0:
            fen_rank += str(empty_count)
        
        fen_ranks.append(fen_rank)
    
    return '/'.join(fen_ranks)


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Load image
        image = Image.open(file.stream).convert('RGB')
        
        # Get predictions
        result = predict_board(image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE)
    })


if __name__ == '__main__':
    print("="*60)
    print("Chess Board Position Classifier - Web App")
    print("="*60)
    
    initialize_model()
    
    print("\nStarting web server...")
    print("Access the app at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
