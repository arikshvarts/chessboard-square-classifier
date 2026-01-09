# How to Transfer Models from Google Colab

**After training completes in Colab, follow these steps to get your models onto your local machine.**

---

## Method 1: Direct Download from Colab (Recommended)

### Step 1: Download from Colab

At the end of your Colab notebook, run this cell:

```python
from google.colab import files

# Download the best model (change fold number to your best)
files.download('dataset_out/best_model_fold_1.pth')

# Or download all fold models
for i in range(1, 8):
 try:
 files.download(f'dataset_out/best_model_fold_{i}.pth')
 except:
 pass
```

This will download files to your computer's Downloads folder.

### Step 2: Place on Local Machine

```bash
# Navigate to your project
cd chessboard-square-classifier

# Copy the best model
# (From Downloads folder to checkpoints/)
cp ~/Downloads/best_model_fold_1.pth checkpoints/best_model.pth

# Windows PowerShell:
Copy-Item "$env:USERPROFILE\Downloads\best_model_fold_1.pth" "checkpoints\best_model.pth"
```

### Step 3: Verify

```bash
python evaluate.py --image test_image.jpg
```

You should see:
```
✓ Model loaded from checkpoints/best_model.pth
 Validation accuracy: XX.XX%
```

---

## Method 2: Via Google Drive (For Large Models)

### Step 1: Save to Drive (Already in Notebook)

Cell 9 in your Colab notebook already does this:

```python
# This saves all models to your Google Drive
!cp -r /content/dataset_out/ /content/drive/MyDrive/chess_models/
```

After running this, your models are in Google Drive at:
```
MyDrive/
└── chess_models/
 └── dataset_out/
 ├── best_model_fold_1.pth
 ├── best_model_fold_2.pth
 └── ...
```

### Step 2: Download from Google Drive

1. Open https://drive.google.com
2. Navigate to `My Drive → chess_models → dataset_out`
3. Right-click `best_model_fold_1.pth` → Download
4. Wait for download (files are ~100MB each)

### Step 3: Place Locally

Same as Method 1 Step 2:

```bash
cd chessboard-square-classifier
cp ~/Downloads/best_model_fold_1.pth checkpoints/best_model.pth
```

---

## Method 3: Create Zip Archive (For Multiple Models)

### In Colab:

```python
# Create zip of all models
!zip -r all_models.zip dataset_out/*.pth

# Download the zip
from google.colab import files
files.download('all_models.zip')
```

### On Local Machine:

```bash
# Extract zip
unzip all_models.zip

# Copy best model
cp dataset_out/best_model_fold_1.pth checkpoints/best_model.pth

# (Optional) Copy all fold models
mkdir -p checkpoints/fold_1
cp dataset_out/best_model_fold_1.pth checkpoints/fold_1/best_model.pth
# Repeat for other folds...
```

---

## Which Fold is Best?

After training completes, Colab will show a summary table:

```
Fold | Test Game | Best Val Acc | Test Acc | Best Epoch
-----|------------------|--------------|----------|------------
1 | game2_per_frame | 94.23% | 92.45% | 6
2 | game4_per_frame | 93.87% | 91.23% | 7
3 | game5_per_frame | 95.12% | 93.78% | 5 ← BEST
...
```

**Choose the fold with highest Test Acc** and use that as your `best_model.pth`.

In the example above, Fold 3 has the best test accuracy (93.78%), so:

```bash
cp dataset_out/best_model_fold_3.pth checkpoints/best_model.pth
```

---

## Expected File Structure

After placing models:

```
checkpoints/
├── best_model.pth # Your best fold (required)
├── README.md # Instructions (already there)
├── .gitignore # Excludes .pth from git
└── fold_1/ # (Optional) Individual folds
 └── best_model.pth
 fold_2/
 └── best_model.pth
 ...
```

**Minimum required**: Just `checkpoints/best_model.pth`

---

## Model File Contents

Each `.pth` file contains:

```python
{
 'model_state_dict': ..., # Model weights
 'optimizer_state_dict': ..., # Optimizer state
 'epoch': 6, # Which epoch
 'val_acc': 94.23, # Validation accuracy
 'fold': 1 # Which fold
}
```

---

## Testing Your Model

### Quick Test:

```bash
python evaluate.py --image test_board.jpg
```

### Expected Output:

```
==============================================================
OFFICIAL EVALUATION API TEST
==============================================================
Image: test_board.jpg
Shape: (1080, 1920, 3)
Dtype: uint8
Range: [0, 255]

Loading checkpoint: checkpoints/best_model.pth
✓ Model loaded from checkpoints/best_model.pth
 Validation accuracy: 94.23%

==============================================================
PREDICTION RESULT
==============================================================
Output type: <class 'torch.Tensor'>
Output shape: torch.Size([8, 8])
Output dtype: torch.int64
Output device: cpu
Value range: [0, 12]

Visualization:
 a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ 7
6 · · · · · · · · 6
5 · · · · · · · · 5
4 · · · · · · · · 4
3 · · · · · · · · 3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
 a b c d e f g h

FEN notation:
 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR

==============================================================
✓ Test complete - function signature is compliant
==============================================================
```

---

## Troubleshooting

### Error: "FileNotFoundError: checkpoints/best_model.pth"

**Solution**: Model not placed correctly. Double-check path:
```bash
ls checkpoints/
# Should show: best_model.pth
```

### Error: "RuntimeError: Error loading checkpoint"

**Possible causes**:
1. **Corrupted download**: Re-download from Colab/Drive
2. **Version mismatch**: Train and eval with same PyTorch version
3. **Incomplete download**: Check file size (should be ~100MB)

**Check file size**:
```bash
ls -lh checkpoints/best_model.pth
# Should be around 95-100 MB
```

### Model Loads But Accuracy is Bad

**If using pretrained (not trained) model**:
- You'll see: "⚠ No trained checkpoint found. Using pretrained ResNet50"
- Accuracy will be poor (random guessing)
- **Solution**: Place your trained model at `checkpoints/best_model.pth`

### Cannot Download from Colab

**If download doesn't start**:
1. Check popup blocker (allow popups from colab.research.google.com)
2. Try Method 2 (Google Drive)
3. Try smaller batch:
 ```python
 # Download one at a time
 files.download('dataset_out/best_model_fold_1.pth')
 ```

---

## For Submission

### Create Download Link for Evaluators

1. **Upload best_model.pth to Google Drive**
2. **Right-click → Share → Anyone with link can view**
3. **Copy link**
4. **Include in README.md**:

```markdown
## Trained Models

Download trained model:
https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing

Place at: `checkpoints/best_model.pth`
```

### Alternative: University Drive

If you have access to university shared drive (up to 2TB):
- Upload there instead
- Provide link in report
- Include placement instructions

---

## Summary

**Simplest workflow**:

1. **In Colab** (after training):
 ```python
 files.download('dataset_out/best_model_fold_3.pth')
 ```

2. **On local machine**:
 ```bash
 cp ~/Downloads/best_model_fold_3.pth checkpoints/best_model.pth
 ```

3. **Test**:
 ```bash
 python evaluate.py --image test.jpg
 ```

4. **Done!** ✓

---

**Need help?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) or [README.md](README.md)
