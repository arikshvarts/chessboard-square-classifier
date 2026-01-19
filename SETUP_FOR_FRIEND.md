# Setup Guide for Running the Chess Web App

**Hey! Here's how to get the web app running with the trained model weights.**

## Quick Start (3 Steps)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd chessboard-square-classifier
```

### 2. Get the Model Weights (CRITICAL - NOT IN GIT)

⚠️ **The model weights are NOT in git** (they're 282MB, too large for git).

You need to get `best_model_fold_1.pth` from Ariel.

**Option A: Download from Google Drive** (recommended)
- Ask Ariel for the Google Drive link
- Download `best_model_fold_1.pth` (282MB)
- Place it in: `checkpoints/best_model_fold_1.pth`

**Option B: Get the file directly**
- Get `best_model_fold_1.pth` from Ariel (via USB, Dropbox, WeTransfer, etc.)
- Place it in: `checkpoints/best_model_fold_1.pth`

### 3. Install & Run
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
```

Open browser to: **http://localhost:5000**

---

## Verifying the Setup

### Check if weights are present:
```bash
# Windows
dir checkpoints\best_model_fold_1.pth

# Mac/Linux
ls -lh checkpoints/best_model_fold_1.pth
```

You should see: `best_model_fold_1.pth` (~282 MB)

### Test the app:
```bash
python app.py
```

Expected output:
```
Using device: cuda  # or cpu
Loading model from: checkpoints/best_model_fold_1.pth
Detected Colab checkpoint format (direct ResNet50)
Model loaded successfully! Val Acc: 98.77
 * Running on http://127.0.0.1:5000
```

---

## Troubleshooting

### ❌ "No checkpoint found"
**Problem:** Model weights not in checkpoints folder  
**Solution:** Get `best_model_fold_1.pth` from Ariel and place in `checkpoints/`

### ❌ "ModuleNotFoundError: No module named 'torch'"
**Problem:** Dependencies not installed  
**Solution:** 
```bash
pip install -r requirements.txt
```

### ❌ "Address already in use"
**Problem:** Port 5000 is taken  
**Solution:** 
```bash
# Run on different port
python app.py --port 5001
```

Or kill the process using port 5000.

### ❌ "CUDA out of memory"
**Problem:** GPU memory full  
**Solution:** App will automatically fallback to CPU (slower but works)

---

## Model Information

**Trained Model:** ResNet50 pretrained on ImageNet
- **Training:** 8 epochs on 4 games (game2, game4, game5, game6)
- **Validation Accuracy:** 98.77%
- **Test Accuracy:** 98.47% (on game2)
- **Classes:** 13 (empty + 12 chess pieces)
- **Input:** 224x224 RGB images
- **Framework:** PyTorch 2.x

---

## File Locations

```
checkpoints/
├── best_model_fold_1.pth    ← YOU NEED THIS FILE (282MB)
├── .gitignore               ← Excludes .pth files from git
└── README.md

app.py                       ← Web application
requirements.txt             ← Dependencies
```

---

## For Ariel: How to Share the Weights

### Option 1: Google Drive (Recommended)
1. Upload `best_model_fold_1.pth` to Google Drive
2. Right-click → Share → Anyone with link can view
3. Copy link and send to friend

### Option 2: Create a Release on GitHub
```bash
# Too large for git, but can use GitHub Releases
# Go to GitHub repository → Releases → Create Release
# Upload best_model_fold_1.pth as asset
```

### Option 3: Cloud Storage
- Dropbox: https://dropbox.com
- WeTransfer: https://wetransfer.com (up to 2GB free)
- OneDrive, Box, etc.

---

## What's Already in Git

✅ **Included in git:**
- All code (app.py, src/, dataset_tools/)
- Configuration files (requirements.txt)
- README and documentation
- Empty checkpoints/ folder structure

❌ **NOT in git (too large):**
- Model weights (.pth files)
- Training data (Data/ folder)
- Dataset outputs (dataset_out/)

---

## Questions?

Contact Ariel or check:
- Main README.md for full documentation
- checkpoints/README.md for model details
- Requirements: Python 3.8+, PyTorch, Flask
