# Chessboard Square Classifier

**Deep Learning course project for chess board position recognition from images.**

**Team**: Ariel Shvarts, Nikol Koifman  
**Course**: Intro to Deep Learning (Fall 2025)  
**Project**: Chessboard Square Classification and Board-State Reconstruction

![Sample debug grid](docs/assets/sample_debug_grid.png)

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Training (Google Colab)](#training-google-colab)
- [Evaluation API](#evaluation-api)
- [Web Application](#web-application)
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
- ✅ 7-fold cross-validation training (train on 6 games, test on 1)
- ✅ ResNet50 CNN architecture with pretrained ImageNet weights
- ✅ Web interface for interactive visualization
- ✅ Official evaluation API: `predict_board(image: np.ndarray) -> torch.Tensor`
- ✅ Compliant dataset format for submission

---

## Repository Structure

```
chessboard-square-classifier/
├── src/                        # Core model code
│   ├── model.py               # ChessSquareClassifier (ResNet50)
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction utilities
│   └── dataset.py             # PyTorch Dataset classes
│
├── dataset_tools/             # Dataset processing utilities
│   ├── make_dataset.py        # Dataset manifest generator
│   ├── extract_squares.py     # Board detection & square extraction
│   ├── fen_utils.py           # FEN notation utilities
│   └── eval.py                # Evaluation metrics
│
├── Data/                      # Raw training data (not in git)
│   ├── game2_per_frame/
│   ├── game4_per_frame/
│   └── ...
│
├── compliant_dataset/         # Converted dataset (generated)
│   ├── images/                # All board images
│   └── gt.csv                 # Ground truth (image_name, FEN, view)
│
├── checkpoints/               # Trained models (download from Colab)
│   ├── best_model.pth         # Best model for evaluation
│   └── fold_*/                # Individual fold models
│
├── templates/                 # Web app HTML
├── static/                    # Web app CSS/JS
│
├── evaluate.py                # ⭐ OFFICIAL EVALUATION API
├── app.py                     # Flask web application
├── create_compliant_dataset.py # Dataset format converter
├── changed.ipynb              # Google Colab training notebook
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd chessboard-square-classifier

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Training Data

Extract `all_games_data.zip` to the `Data/` folder:

```powershell
Expand-Archive -Path all_games_data.zip -DestinationPath .
```

Your structure should look like:
```
Data/
├── game2_per_frame/
│   ├── tagged_images/
│   └── game2.csv
├── game4_per_frame/
└── ...
```

### 3. Convert to Compliant Format

```bash
python create_compliant_dataset.py --input Data --output compliant_dataset
```

This creates:
- `compliant_dataset/images/` - All board images
- `compliant_dataset/gt.csv` - Ground truth (image_name, FEN, view)

### 4. Download Trained Models

After training in Colab (see next section), download models to `checkpoints/`:

```
checkpoints/
└── best_model.pth    # Place your best model here
```

See [checkpoints/README.md](checkpoints/README.md) for detailed instructions.

---

## Training (Google Colab)

Training is done in Google Colab for GPU access. Use the provided notebook:

### Step 1: Open Notebook in Colab

Upload `changed.ipynb` to Google Colab or open directly from GitHub.

### Step 2: Prepare Files

Create two zip files:
1. **code.zip** - Contains `src/` and `dataset_tools/` folders
2. **all_games_data.zip** - Your training data

```bash
# Create code.zip
zip -r code.zip src/ dataset_tools/
```

### Step 3: Run Notebook Cells

Execute cells in order:

1. **Cell 1**: Check PyTorch/CUDA
2. **Cell 2**: Install packages
3. **Cell 3**: Upload `code.zip`
4. **Cell 4**: Upload `all_games_data.zip`
5. **Cell 5**: Documentation (skip)
6. **Cell 6**: Prepare 7-fold splits (~1 min)
7. **Cell 7**: **Train all 7 folds** (~2-3 hours with GPU)
8. **Cell 8**: Visualize results
9. **Cell 9**: **Save to Google Drive** (important!)
10. **Cell 10**: (Optional) Detailed analysis

### Step 4: Download Models

After training completes:

```python
# In Colab:
from google.colab import files
files.download('dataset_out/best_model_fold_1.pth')
```

Or download from Google Drive: `MyDrive/chess_models/`

### Step 5: Place Models Locally

Copy the best model to `checkpoints/best_model.pth` on your local machine.

**Training Configuration:**
- Model: ResNet50 (pretrained on ImageNet)
- Epochs: 8 per fold
- Batch Size: 128
- Optimizer: Adam (lr=0.001)
- Data Augmentation: RandomHorizontalFlip, ColorJitter

---

## Evaluation API

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

### Class Encoding (Official Spec)

```
0: White Pawn    | 6: Black Pawn
1: White Rook    | 7: Black Rook
2: White Knight  | 8: Black Knight
3: White Bishop  | 9: Black Bishop
4: White Queen   | 10: Black Queen
5: White King    | 11: Black King
12: Empty Square
13: Out-of-Distribution / Unknown
```

### Usage Example

```python
from evaluate import predict_board
import numpy as np
from PIL import Image

# Load image as numpy array
image = np.array(Image.open('chessboard.jpg').convert('RGB'))

# Get prediction
board_tensor = predict_board(image)  # Shape: (8, 8)

print(board_tensor)
# tensor([[7, 8, 9, 10, 11, ...],
#         [6, 6, 6, 6, 6, ...],
#         ...])
```

### Test Evaluation API

```bash
python evaluate.py --image path/to/chessboard.jpg --show-tensor
```

---

## Web Application

Interactive web interface for visualizing predictions.

### Start Web Server

```bash
# Make sure you're in the project directory
cd chessboard-square-classifier

# Start Flask app
python app.py
```

Access at: **http://localhost:5000**

### Features

- 📤 Drag-and-drop image upload
- ♟️ Visual chess board display
- 📊 Confidence scores per square
- 📋 FEN notation output
- 📈 Prediction statistics

### Screenshot

The web app shows:
1. Upload area
2. Detected chess board (8×8 grid)
3. FEN notation
4. Confidence analysis
5. Statistics panel

---

## Dataset Format

### Raw Format (As Collected)

```
Data/
├── game2_per_frame/
│   ├── tagged_images/
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── game2.csv (columns: from_frame, fen)
└── ...
```

### Compliant Format (For Submission)

```
compliant_dataset/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── gt.csv
```

**gt.csv format:**
```csv
image_name,fen,view
frame_000001.jpg,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR,white_bottom
frame_000002.jpg,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR,white_bottom
...
```

### Convert Dataset

```bash
python create_compliant_dataset.py --input Data --output compliant_dataset

# Verify format
python create_compliant_dataset.py --output compliant_dataset --verify
```

---

## Requirements

### Python Version
- Python 3.8+

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
opencv-python>=4.7.0
python-chess>=1.9.0
tqdm>=4.65.0
flask>=2.3.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Additional Information

### Dataset Source
- **Provided by**: Course instructors
- **Content**: Real chess games with ground truth FEN positions
- **Games**: 7 games with varying conditions
- **Total Frames**: ~300-500 frames per game
- **Annotations**: FEN notation for each frame

### Model Performance
- **Cross-validation**: 7-fold (train on 6 games, test on 1)
- **Expected accuracy**: 90-95% (varies by game)
- **Training time**: ~2-3 hours on Colab GPU

### Links
- Lichess Database: https://database.lichess.org/
- FEN Notation: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation

---

## Troubleshooting

### Model Not Loading
- Ensure `checkpoints/best_model.pth` exists
- Check file isn't corrupted
- Verify trained with same PyTorch version

### Board Detection Fails
- Image must show clear chessboard
- Try different lighting/angle
- Check `extract_squares.py` settings

### Web App Not Starting
- Check port 5000 is not in use
- Ensure Flask is installed
- Try: `python app.py` from project root

---

## Contact

**Team Members:**
- Ariel Shvarts
- Nikol Koifman

**Course:** Intro to Deep Learning (Fall 2025)

---

## License

This project is for educational purposes as part of a university course.
- Kaggle Chess Piece Images dataset: [https://www.kaggle.com/datasets/koryakinp/chess-pieces-images](https://www.kaggle.com/datasets/koryakinp/chess-positions)
- https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2

## Evaluation (later)
- Compare predictions vs manifest: `python dataset_tools/eval.py --manifest dataset_out/dataset_manifest.csv --preds path/to/preds.csv`

## Git workflow
- Start: `git checkout -b feature/<task>`
- Sync with base: `git checkout main` → `git pull` → `git checkout feature/<task>` → `git merge main`
- Commit/push: `git add ...` → `git commit -m "..."` → `git push -u origin feature/<task>`
- Open PR: base = `main`, compare = `feature/<task>`; request review; merge when green.
- After merge: `git checkout main` → `git pull` → `git branch -d feature/<task>` → `git push origin --delete feature/<task>`
- Keep data out of Git: `Data/`, `dataset_out/`, `checkpoints/`, `outputs/` stay in `.gitignore`.


