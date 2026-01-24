# Chessboard Square Classifier

**Deep Learning course project for chess board position recognition from images.**

**Team**: Ariel Shvarts, Nikol Koifman, Yaakov Gerelter
**Course**: Intro to Deep Learning (Fall 2025)
**Project**: Chessboard Square Classification and Board-State Reconstruction

**🌐 Live Demo**: https://chessboard-square-classifier-iardjdxpqbydrrvjbyhqoy.streamlit.app/

![Sample debug grid](docs/assets/sample_debug_grid.png)

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Live Demo](#live-demo)
- [Model Training](#model-training)
- [Evaluation API](#evaluation-api)
- [Dataset Format](#dataset-format)
- [Requirements](#requirements)

---

## Overview

This project implements a chess board position classifier that:
1. Detects and extracts 64 squares from a chessboard image
2. Classifies each square (empty or one of 12 piece types)
3. Reconstructs the full board state as FEN notation
4. Provides evaluation API compliant with course specifications

**Key Features:**
- 7-fold cross-validation training (train on 6 games, test on 1)
- ResNet50 CNN architecture with pretrained ImageNet weights
- Interactive web demo (Streamlit Cloud)
- Official evaluation API: `predict_board(image: np.ndarray) -> torch.Tensor`
- Compliant dataset format for submission
- Auto-download model from Google Drive

---

## Repository Structure

```
chessboard-square-classifier/
├── src/ # Core model code
│ ├── model.py # ChessSquareClassifier (ResNet50)
│ ├── train.py # Training script
│ ├── predict.py # Prediction utilities
│ └── dataset.py # PyTorch Dataset classes
│
├── dataset_tools/ # Dataset processing utilities
│ ├── make_dataset.py # Dataset manifest generator
│ ├── extract_squares.py # Board detection & square extraction
│ ├── fen_utils.py # FEN notation utilities
│ └── eval.py # Evaluation metrics
│
├── Data/ # Raw training data (not in git)
│ ├── game2_per_frame/
│ ├── game4_per_frame/
│ └── ...
│
├── compliant_dataset/ # Converted dataset (generated)
│ ├── images/ # All board images
│ └── gt.csv # Ground truth (image_name, FEN, view)
│
├── checkpoints/ # Trained models
│ ├── best_model.pth # Best model for evaluation
│ └── README.md # Model download instructions
│
├── evaluate.py # OFFICIAL EVALUATION API
├── streamlit_app.py # Streamlit web application
├── demo.py # Demo script
├── requirements.txt # Python dependencies
└── README.md # This file
```

---

## Quick Start

### For Evaluators

1. **Live Demo**: https://chessboard-square-classifier-iardjdxpqbydrrvjbyhqoy.streamlit.app/
2. **Model Weights**: https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9?usp=drive_link
3. **Evaluation API**: See [Evaluation API](#evaluation-api) section below

### Setup (Optional - for local testing)

```bash
# Clone repository
git clone https://github.com/arikshvarts/chessboard-square-classifier.git
cd chessboard-square-classifier

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download model from Google Drive link above and place in checkpoints/
```

---

## Live Demo

**Download the trained model from Google Drive:**

🔗 **[Download Model Weights](https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9?usp=drive_link)**

After downloading, place `best_model.pth` in the `checkpoints/` folder:

```
checkpoints/
└── best_model.pth # Place downloaded model here
```

See [checkpoints/README.md](checkpoints/README.md) for detailed instructions.

---

## Model Training

**Trained Model**: https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9?usp=drive_link

The model was trained using 7-fold cross-validation on Google Colab.

**Training Configuration:**
- **Model**: ResNet50 (pretrained on ImageNet)
- **Epochs**: 8 per fold
- **Batch Size**: 128
- **Optimizer**: Adam (lr=0.001)
- **Data Augmentation**: ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1), RandomRotation(±5°)
- **Cross-validation**: 7-fold (train on 6 games, test on 1)
- **Training time**: ~2-3 hours on Colab GPU
- **Final Accuracy**: ~98.8%

---

## Evaluation API

**FOR EVALUATORS - START HERE**


This section describes the official evaluation function required for grading, following the instructor's exact specifications.

### Official Function Signature

```python
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict chessboard state from RGB image.
    
    Args:
        image: numpy.ndarray, shape (H, W, 3), RGB, uint8, [0-255]
        
    Returns:
        torch.Tensor, shape (8, 8), dtype torch.int64, device CPU
        Values: 0-11 (pieces), 12 (empty), 13 (OOD/unknown)
    """
```

### Input Specification
- **Type**: `numpy.ndarray`
- **Shape**: `(H, W, 3)`
- **Channel order**: RGB
- **Dtype**: `uint8`
- **Value range**: [0, 255]

### Output Specification
- **Type**: `torch.Tensor`
- **Shape**: `(8, 8)`
- **Device**: CPU
- **Dtype**: `torch.int64`
- **Values**: Integers in range [0, 13]

### Board Coordinate Convention
**IMPORTANT**: The mapping is purely image-based, NOT chess notation based.

- `output[0, 0]` → top-left square of the **image**
- `output[0, 7]` → top-right square of the **image**
- `output[7, 0]` → bottom-left square of the **image**
- `output[7, 7]` → bottom-right square of the **image**

### Class Encoding (Official Spec)

| Value | Class | Value | Class |
|-------|-------|-------|-------|
| 0 | White Pawn | 6 | Black Pawn |
| 1 | White Rook | 7 | Black Rook |
| 2 | White Knight | 8 | Black Knight |
| 3 | White Bishop | 9 | Black Bishop |
| 4 | White Queen | 10 | Black Queen |
| 5 | White King | 11 | Black King |
| 12 | Empty Square | 13 | OOD / Unknown |

### OOD (Out-of-Distribution) Handling

Value `13` is used for squares that:
- Do not contain a valid chess piece
- Are occluded or ambiguous  
- Contain objects not belonging to the 12 known classes
- Are not clearly empty squares

**Output Visualization**: For Project 1, OOD squares are marked with red X in `./results/` folder.

### Usage Example

```python
from evaluate import predict_board
import numpy as np
from PIL import Image

# Load image as numpy array
image = np.array(Image.open('chessboard.jpg').convert('RGB'))

# Get prediction
board_tensor = predict_board(image) # Shape: (8, 8)

print(board_tensor)
# tensor([[7, 8, 9, 10, 11, ...],
# [6, 6, 6, 6, 6, ...],
# ...])
```

### Test Evaluation API

Basic usage:
```bash
python evaluate.py --image path/to/chessboard.jpg --show-tensor
```

With OOD visualization (Project 1 requirement):
```bash
python evaluate.py --image path/to/chessboard.jpg --save-viz --output ./results/prediction.png
```

This will:
1. Call `predict_board(image)` function
2. Display the predicted 8x8 tensor
3. Save visualization with red X marks on OOD squares to `./results/` folder

---

## Live Demo

**🌐 Try it now**: https://chessboard-square-classifier-iardjdxpqbydrrvjbyhqoy.streamlit.app/

Interactive web application deployed on Streamlit Cloud.

### Features

- 📤 Upload chess board images
- 🎯 Real-time predictions with ResNet50 model
- Visual board display with piece labels
- Confidence scores per square
- FEN notation output
- Automatic model download from Google Drive

### Run Locally (Optional)

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

Access at: **http://localhost:8501**

---

## Dataset Format

### Understanding Dataset Formats

This project works with three different dataset formats:

#### Format 1: Original Raw Data (As Provided by Instructor)

```
Data/
├── game2_per_frame/
│   ├── tagged_images/
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── game2.csv (columns: from_frame, fen)
├── game4_per_frame/
└── ...
```

**Used for**: Local training with `src/train.py`  
**CSV format**: Each game has its own CSV with `from_frame` and `fen` columns

#### Format 2: Compliant Dataset (Required for Submission)

```
compliant_dataset/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── gt.csv
```

**Used for**: Submission to instructor (Google Drive upload)  
**CSV format**: Single `gt.csv` with columns: `image_name`, `fen`, `view`

Example `gt.csv`:
```csv
image_name,fen,view
frame_000001.jpg,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR,white_bottom
frame_000002.jpg,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR,white_bottom
```

#### Format 3: Training Manifest (Generated by Preprocessing)

```
dataset_out/
├── dataset_manifest.csv
└── classes.json
```

**Used for**: Training with `src/train.py` after preprocessing  
**CSV format**: Manifest with columns: `image_path`, `label_0` to `label_63`, `split`, `game_id`

---

### Which Format to Use?

| Task | Format | Command |
|------|--------|---------|
| **Local Training** | Format 1 (Original) | `python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out`<br>`python src/train.py --manifest dataset_out/dataset_manifest.csv` |
| **Google Drive Submission** | Format 2 (Compliant) | `python create_compliant_dataset.py --input Data --output compliant_dataset` |
| **Evaluation with trained model** | Any format | `python evaluate.py --image <path>` or `python demo.py --image <path>` |

---

### Converting Between Formats

#### Raw Data → Compliant Format (for submission)

```bash
python create_compliant_dataset.py --input Data --output compliant_dataset

# Verify the output
python create_compliant_dataset.py --output compliant_dataset --verify
```

Upload `compliant_dataset/` folder to Google Drive for submission.

#### Raw Data → Training Manifest (for local training)

```bash
python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out
```

This generates the manifest needed for `src/train.py`.

---

## Demo Script

Quick demonstration of the prediction function:

```bash
# Run demo on a single image
python demo.py --image path/to/chessboard.jpg

# Save results to custom directory
python demo.py --image examples/board1.jpg --output demo_results/
```

The demo script shows:
- Predicted board in readable format
- Class IDs for each square
- Visual board layout (8x8 grid)
- Results saved to text file

---

## Requirements

### Python Version
- Python 3.8+

### Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Contact

**Team**: Ariel Shvarts, Nikol Koifman, Yaakov Gerelter  
**Course**: Intro to Deep Learning (Fall 2025)

---

## License

This project is for educational purposes as part of a university course.


