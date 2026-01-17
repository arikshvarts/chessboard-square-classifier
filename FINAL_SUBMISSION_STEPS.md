# Final Submission Instructions - Step by Step

## ✅ What Has Been Done

### Code Compliance
- ✅ **evaluate.py** - `predict_board()` function matches instructor specification exactly
- ✅ **Input**: numpy.ndarray (H, W, 3), RGB, uint8, [0-255]
- ✅ **Output**: torch.Tensor (8, 8), CPU, int64, values [0-13]
- ✅ **Coordinate convention**: output[0,0] = top-left of IMAGE (not chess notation)
- ✅ **OOD handling**: Value 13 for unknown/invalid squares
- ✅ **Visualization**: Red X marks on OOD squares saved to `./results/` folder

### Dataset Format (Compliant)
- ✅ `create_compliant_dataset.py` creates correct structure:
  ```
  compliant_dataset/
  ├── images/           # All frame images
  └── gt.csv            # 3 columns: image_name, fen, view
  ```
- ✅ View specification: "white_bottom" or "black_bottom"

### Documentation
- ✅ README.md with complete instructions
- ✅ requirements.txt with all dependencies
- ✅ Python version specified (3.8+)
- ✅ Training instructions (local and Colab)
- ✅ Demo script (demo.py)
- ✅ Cleaned up unnecessary files

---

## 📋 STEP-BY-STEP: What YOU Must Do Now

### STEP 1: Test Everything Works

#### 1a. Test Evaluation API
```bash
python evaluate.py --image <test_image.jpg> --save-viz
```

Check that:
- Prediction completes without errors
- Output tensor is (8, 8), dtype int64
- File saved to `./results/prediction.png` with red X on OOD squares

#### 1b. Test Demo Script
```bash
python demo.py --image <test_image.jpg>
```

Verify the board visualization displays correctly.

#### 1c. Test Training (Optional)
```bash
# Generate training manifest
python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out

# Test 1 epoch
python src/train.py --manifest dataset_out/dataset_manifest.csv --epochs 1
```

---

### STEP 2: Prepare Dataset for Submission

#### 2a. Generate Compliant Dataset
```bash
python create_compliant_dataset.py --input Data --output compliant_dataset
```

This creates:
```
compliant_dataset/
├── images/              # All frame images
└── gt.csv               # 3 columns: image_name, fen, view
```

#### 2b. Verify Dataset Format
```bash
python create_compliant_dataset.py --output compliant_dataset --verify
```

Check output shows:
- ✅ `images/` folder exists
- ✅ `gt.csv` exists with exactly 3 columns
- ✅ All images in CSV exist in images/ folder

#### 2c. Verify gt.csv Structure
Open `compliant_dataset/gt.csv` and verify:
```csv
image_name,fen,view
frame_000001.jpg,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR,white_bottom
frame_000002.jpg,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR,white_bottom
```

**Columns must be**:
1. `image_name` - filename.jpg
2. `fen` - FEN string
3. `view` - "white_bottom" or "black_bottom"

---

### STEP 3: Upload to Google Drive

#### Required Uploads:
1. **compliant_dataset/** folder (REQUIRED format)
2. **Data/** folder (original format - if you used different format for training)
3. **checkpoints/best_model.pth** (trained model weights)

#### Upload Steps:
```bash
# On Google Drive:
1. Create folder: "Chess_Project_Final_Submission"
2. Upload: compliant_dataset/ (zip it first if large)
3. Upload: Data/ (original data, if different from compliant)
4. Upload: checkpoints/best_model.pth
5. Set sharing: "Anyone with link can view"
6. Copy shareable links
```

---

### STEP 4: Git Cleanup and Push

#### 4a. Check Status
```bash
cd "c:\Users\ariks\uni\DeepLearning\Final_miss_clone_for_web_App\chessboard-square-classifier"
git status
```

#### 4b. Add Files
```bash
git add .gitignore
git add evaluate.py
git add demo.py
git add README.md
git add SUBMISSION_CHECKLIST.md
git add FINAL_SUBMISSION_STEPS.md
git add requirements.txt
git add src/
git add dataset_tools/
git add templates/
git add static/
git add app.py
git add create_compliant_dataset.py
```

#### 4c. Commit
```bash
git commit -m "Final submission: evaluation API compliant, dataset format correct, OOD visualization added"
```

#### 4d. Push
```bash
git push origin main
```

---

### STEP 5: Create Final Report

Create a PDF report (max 25 pages) with:

1. **Abstract** - Problem, approach, results
2. **Introduction** - Task description, challenges, contributions
3. **Method** - Model architecture (ResNet50), training procedure, data augmentation
4. **Experiments**:
   - Dataset: 7 games, ~300-500 frames each
   - 7-fold cross-validation
   - Metrics: accuracy, per-class precision/recall
   - Results tables and visualizations
5. **Ablation Study** (REQUIRED):
   - Remove data augmentation → what happens?
   - Use ResNet18 vs ResNet50 → comparison?
   - Different preprocessing → results?
6. **Discussion** - Failure cases, limitations
7. **References** - Papers, datasets, repositories used

---

### STEP 6: Prepare Presentation (7-10 minutes)

Create slides with:

1. **Introduction** (1 slide)
   - Your names and degrees
   - Project: Chessboard Square Classification

2. **Problem** (1 slide)
   - Input: Chess board image
   - Output: 8x8 tensor with piece classifications

3. **Method** (2-3 slides)
   - ResNet50 architecture
   - 7-fold cross-validation
   - Data augmentation

4. **What's Special** (1 slide)
   - Your unique approach or insight

5. **Results** (2-3 slides)
   - Accuracy numbers
   - Visual examples
   - Ablation results

6. **Learnings** (1 slide - optional)
   - Challenges encountered
   - Insights gained

---

### STEP 7: Final Checklist

Before submission on January 24, 2026:

#### Code Repository (GitHub):
- ✅ All source code pushed
- ✅ requirements.txt present
- ✅ README.md with complete instructions
- ✅ evaluate.py with compliant `predict_board()` function
- ✅ demo.py for demonstration
- ✅ No zip files or PDFs in repo

#### Google Drive:
- ✅ compliant_dataset/ folder uploaded (images/ + gt.csv)
- ✅ Data/ folder uploaded (if different from compliant)
- ✅ checkpoints/best_model.pth uploaded
- ✅ Sharing links accessible

#### Report:
- ✅ PDF created (max 25 pages)
- ✅ All sections complete
- ✅ Ablation study included
- ✅ References listed

#### Presentation:
- ✅ Slides ready (7-10 minutes)
- ✅ Practice timing
- ✅ Visual examples included

---

## 🎯 Critical Requirements Verification

### Evaluation API (MUST MATCH EXACTLY):

```python
def predict_board(image: np.ndarray) -> torch.Tensor:
    # Input: (H, W, 3), RGB, uint8, [0-255]
    # Output: (8, 8), CPU, int64, [0-13]
    # output[0,0] = top-left of IMAGE
    # 13 = OOD/unknown
```

### Dataset Format (MUST MATCH EXACTLY):

```
compliant_dataset/
├── images/
│   ├── frame_000001.jpg
│   └── ...
└── gt.csv  (3 columns: image_name, fen, view)
```

### Output Visualization (Project 1):

- OOD squares (value 13) marked with red X
- Saved to `./results/` folder

---

## 📞 If You Need Help

**Issues?** Check:
1. README.md - Complete instructions
2. SUBMISSION_CHECKLIST.md - Quick reference
3. Run: `python evaluate.py --help`
4. Run: `python demo.py --help`

**Instructor Office Hours**: Check course announcements

---

## 🚀 Quick Commands Reference

```bash
# Test evaluation
python evaluate.py --image test.jpg --save-viz

# Test demo
python demo.py --image test.jpg

# Generate compliant dataset
python create_compliant_dataset.py --input Data --output compliant_dataset

# Test training
python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out
python src/train.py --manifest dataset_out/dataset_manifest.csv --epochs 1

# Git push
git add .
git commit -m "Final submission"
git push origin main
```

---

**Deadline: January 24, 2026**

Good luck! 🍀
