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
- [Training (Google Colab)](#training-google-colab)
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
├── checkpoints/ # Trained models (download from Colab)
│ ├── best_model.pth # REQUIRED: Best model for evaluation
│ └── fold_*/ # Individual fold models
│
├── templates/ # Web app HTML (Streamlit)
├── static/ # Web app CSS/JS (legacy)
│
├── evaluate.py # OFFICIAL EVALUATION API
├── streamlit_app.py # Streamlit web application
├── app.py # Flask web application (legacy)
├── create_compliant_dataset.py # Dataset format converter
├── changed.ipynb # Google Colab training notebook
├── requirements.txt # Python dependencies
└── README.md # This file
```

---

## Quick Start

**For Evaluators**: Jump to [Evaluation API](#evaluation-api) section.

**For Students/Development**:

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
│ ├── tagged_images/
│ └── game2.csv
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

**Download the trained model from Google Drive:**

🔗 **[Download Model Weights](https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9?usp=drive_link)**

After downloading, place `best_model.pth` in the `checkpoints/` folder:

```
checkpoints/
└── best_model.pth # Place downloaded model here
```

See [checkpoints/README.md](checkpoints/README.md) for detailed instructions.

---

## Training (For Students Only - Not Required for Evaluation)

**Note for Evaluators**: You do NOT need to train the model. Skip to [Evaluation API](#evaluation-api) section. The trained model will be provided.

---

### Option 1: Training in Google Colab (Recommended)

Training is done in Google Colab for GPU access. Use the provided notebook:

#### Step 1: Open Notebook in Colab

Upload `changed.ipynb` to Google Colab or open directly from GitHub.

#### Step 2: Prepare Files

Create two zip files:
1. **code.zip** - Contains `src/` and `dataset_tools/` folders
2. **all_games_data.zip** - Your training data

```bash
# Create code.zip
zip -r code.zip src/ dataset_tools/
```

#### Step 3: Run Notebook Cells

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

#### Step 4: Download Models

After training completes:

```python
# In Colab:
from google.colab import files
files.download('dataset_out/best_model_fold_1.pth')
```

Or download from Google Drive: `MyDrive/chess_models/`

#### Step 5: Place Models Locally

Copy the best model to `checkpoints/best_model.pth` on your local machine.

**Training Configuration:**
- Model: ResNet50 (pretrained on ImageNet)
- Epochs: 8 per fold
- Batch Size: 128
- Optimizer: Adam (lr=0.001)
- Data Augmentation: RandomHorizontalFlip, ColorJitter

---

### Option 2: Training Locally

#### Prerequisites
- Python 3.8+
- GPU recommended (CUDA-enabled PyTorch)
- 8GB+ RAM
- 10GB+ disk space

#### Step 1: Data Placement

Place your raw training data in the `Data/` directory:

```
Data/
├── game2_per_frame/
│   ├── tagged_images/
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── game2.csv
├── game4_per_frame/
├── game5_per_frame/
├── game6_per_frame/
├── game7_per_frame/
├── game8_per_frame/
└── game9_per_frame/
```

Each game folder must contain:
- `tagged_images/` - Folder with frame images
- `<game_name>.csv` - CSV with columns: `from_frame`, `fen`

#### Step 2: Preprocessing - Generate Dataset Manifest

Run the dataset preparation script to create training manifest:

```bash
python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out
```

This creates:
- `dataset_out/dataset_manifest.csv` - Manifest with all training samples
- `dataset_out/classes.json` - Class label mappings

The manifest contains columns: `image_path`, `label_<square>` (64 columns), `split`, `game_id`

#### Step 3: Train the Model

Run training with default parameters:

```bash
python src/train.py --manifest dataset_out/dataset_manifest.csv --classes dataset_out/classes.json --epochs 20 --batch_size 64 --output_dir checkpoints
```

Training parameters:
- `--manifest`: Path to dataset manifest CSV (from Step 2)
- `--classes`: Path to classes JSON (from Step 2)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: DataLoader workers (default: 4)
- `--output_dir`: Where to save checkpoints (default: checkpoints)

#### Step 4: Monitor Training

Training will display progress bars and metrics:
```
Epoch 1/20 [Train]: 100%|████| 150/150 [02:30<00:00]
Train Loss: 0.4523 | Train Acc: 88.32%
Epoch 1/20 [Val]: 100%|████| 25/25 [00:15<00:00]
Val Loss: 0.3214 | Val Acc: 91.45%
New best model! Val Acc: 91.45%
```

Models saved:
- `checkpoints/best_model.pth` - Best validation accuracy
- `checkpoints/latest_model.pth` - Latest checkpoint
- `checkpoints/training_history.json` - Loss/accuracy history

#### Step 5: Use Trained Model

The best model will be automatically saved to `checkpoints/best_model.pth` and can be used with `evaluate.py` or `demo.py`.

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


