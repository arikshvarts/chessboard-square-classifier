"""
Chess Board Position Classifier - Streamlit App
Deploy: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_tools'))

from model import ChessSquareClassifier
from extract_squares import extract_64_square_crops
from torchvision import transforms

# Page config
st.set_page_config(
    page_title="Chess Board Classifier",
    page_icon="♟️",
    layout="wide"
)

# Class mapping
INTERNAL_TO_SPEC = {
    0: 12, 1: 0, 2: 2, 3: 3, 4: 1, 5: 4, 6: 5,
    7: 6, 8: 8, 9: 9, 10: 7, 11: 10, 12: 11
}

PIECE_NAMES = {
    0: '♙ (White Pawn)', 1: '♖ (White Rook)', 2: '♘ (White Knight)',
    3: '♗ (White Bishop)', 4: '♕ (White Queen)', 5: '♔ (White King)',
    6: '♟ (Black Pawn)', 7: '♜ (Black Rook)', 8: '♞ (Black Knight)',
    9: '♝ (Black Bishop)', 10: '♛ (Black Queen)', 11: '♚ (Black King)',
    12: '⬜ (Empty)', 13: '❌ (OOD)'
}

def download_model_from_gdrive():
    """Download model from Google Drive if not present."""
    import gdown
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive (this may take a few minutes - ~282MB)...")
        try:
            # Download from Google Drive folder
            # Using folder ID from: https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9
            folder_url = "https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9"
            
            # Download the folder contents to checkpoints/
            gdown.download_folder(folder_url, output=checkpoint_dir, quiet=False, use_cookies=False)
            
            # Check if model was downloaded
            if os.path.exists(model_path):
                st.success("✅ Model downloaded successfully!")
            else:
                # Try alternative file name
                alt_path = os.path.join(checkpoint_dir, 'best_model_fold_1.pth')
                if os.path.exists(alt_path):
                    model_path = alt_path
                    st.success("✅ Model downloaded successfully!")
                else:
                    st.error("❌ Model file not found after download")
                    return None
                    
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            st.info("💡 The model will try to load from local checkpoints/ folder if available.")
            st.markdown("**For evaluators:** Please ensure the Google Drive folder has public 'Anyone with link can view' access.")
            return None
    
    return model_path


@st.cache_resource
def load_model():
    """Load the trained model once."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try to download model from Google Drive if not present
    model_path = download_model_from_gdrive()
    
    if model_path is None:
        # Fallback to local paths
        model_paths = [
            'checkpoints/best_model.pth',
            'checkpoints/best_model_fold_1.pth'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        st.error("❌ Model file not found! Please check Google Drive link.")
        return None, None
    
    try:
        # Load checkpoint (weights_only=False since this is our own trusted model)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            # Direct state dict
            state_dict = checkpoint
        
        # Check if checkpoint is from Colab (direct ResNet50) or local (ChessSquareClassifier wrapper)
        if 'backbone.conv1.weight' not in state_dict and 'conv1.weight' in state_dict:
            # Colab checkpoint - load directly into ResNet50
            from torchvision import models
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 13)
            model.load_state_dict(state_dict)
        else:
            # Local checkpoint - use ChessSquareClassifier wrapper
            model = ChessSquareClassifier(num_classes=13)
            model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        return None, None

def get_transform():
    """Get inference transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_board(image, model, device, transform):
    """Predict chess board position."""
    # Extract 64 squares
    squares = extract_64_square_crops(image)
    
    # Predict each square
    predictions = []
    with torch.no_grad():
        for square in squares:
            square_tensor = transform(square).unsqueeze(0).to(device)
            output = model(square_tensor)
            pred = output.argmax(dim=1).item()
            # Convert to spec encoding
            spec_pred = INTERNAL_TO_SPEC.get(pred, 13)
            predictions.append(spec_pred)
    
    # Reshape to 8x8 board
    board = torch.tensor(predictions, dtype=torch.int64).reshape(8, 8)
    return board

def visualize_board(board_tensor):
    """Create a visual representation of the board."""
    board_html = "<div style='font-family: monospace; font-size: 14px;'>"
    
    for row in range(8):
        board_html += "<div style='margin: 2px 0;'>"
        for col in range(8):
            val = board_tensor[row, col].item()
            piece = PIECE_NAMES.get(val, '?')
            
            # Color coding
            if val == 13:
                color = '#ff0000'  # Red for OOD
            elif val == 12:
                color = '#cccccc'  # Gray for empty
            elif val <= 5:
                color = '#3498db'  # Blue for white pieces
            else:
                color = '#2c3e50'  # Dark for black pieces
            
            board_html += f"<span style='color: {color}; padding: 5px;'>{piece}</span>"
        board_html += "</div>"
    
    board_html += "</div>"
    return board_html

def create_board_image(board_tensor, original_image):
    """Create annotated board image with predictions."""
    img = original_image.copy()
    draw = ImageDraw.Draw(img)
    
    w, h = img.size
    sq_w, sq_h = w // 8, h // 8
    
    for row in range(8):
        for col in range(8):
            val = board_tensor[row, col].item()
            
            # Draw red X for OOD
            if val == 13:
                left = col * sq_w
                top = row * sq_h
                right = (col + 1) * sq_w
                bottom = (row + 1) * sq_h
                
                draw.line([(left, top), (right, bottom)], fill='red', width=3)
                draw.line([(left, bottom), (right, top)], fill='red', width=3)
    
    return img

# Title
st.title("♟️ Chess Board Position Classifier")
st.markdown("Upload a chess board image to classify each square")

# Load model
with st.spinner("Loading model..."):
    model, device = load_model()

if model is None:
    st.stop()

st.success(f"✅ Model loaded on {device}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a chess board image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a square image of a chess board"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Predict
    with st.spinner("Analyzing board..."):
        transform = get_transform()
        board_tensor = predict_board(image, model, device, transform)
    
    with col2:
        st.subheader("Detected Position")
        
        # Count pieces
        piece_counts = {}
        ood_count = 0
        for val in board_tensor.flatten().tolist():
            if val == 13:
                ood_count += 1
            else:
                piece_counts[val] = piece_counts.get(val, 0) + 1
        
        # Display board
        board_html = visualize_board(board_tensor)
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Stats
        st.markdown("---")
        if ood_count > 0:
            st.warning(f"⚠️ {ood_count} Out-of-Distribution squares detected")
        
        # Show annotated image
        annotated = create_board_image(board_tensor, image)
        st.image(annotated, caption="Annotated (Red X = OOD)", use_container_width=True)
    
    # Download results
    st.markdown("---")
    st.subheader("Export Results")
    
    # Convert to numpy for download
    board_np = board_tensor.numpy()
    st.download_button(
        label="📥 Download Board Tensor (CSV)",
        data=np.array2string(board_np, separator=','),
        file_name="board_prediction.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a chess board image to get started")
    
    # Show example info
    st.markdown("---")
    st.markdown("### Expected Input")
    st.markdown("""
    - Image should be a square crop of the chess board
    - Board should be oriented with white pieces at the bottom (or specify view)
    - Image quality: 800x800 pixels or higher recommended
    """)

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **Deep Learning Chess Classifier**
    
    - Model: ResNet50
    - Classes: 13 (12 pieces + empty + OOD)
    - Training: 7-fold cross-validation
    """)
    
    st.markdown("---")
    st.markdown("### Class Encoding")
    for idx, name in PIECE_NAMES.items():
        st.markdown(f"`{idx:2d}` → {name}")
