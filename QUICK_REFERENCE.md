# Quick Reference Guide

**Everything you need to know in one place.**

---

## 🎯 What I Changed

### 1. **Created evaluate.py** - Official Evaluation API
**Location**: `evaluate.py` (root directory)

**What it does**:
- Provides the EXACT function required by course spec
- Signature: `predict_board(image: np.ndarray) -> torch.Tensor`
- Handles encoding conversion automatically (internal → spec)
- Loads your trained model from `checkpoints/best_model.pth`

**Test it**:
```bash
python evaluate.py --image path/to/chessboard.jpg
```

### 2. **Updated app.py** - Web App Still Works
**Location**: `app.py`

**What I did**:
- Kept web app fully functional for visualization
- Uses internal encoding (0=empty, 1-12=pieces)
- Added comments explaining encoding system
- No breaking changes - everything works as before

**Run it**:
```bash
python app.py
# Visit http://localhost:5000
```

### 3. **Created create_compliant_dataset.py** - Dataset Converter
**Location**: `create_compliant_dataset.py`

**What it does**:
- Converts your Data/ folder to required format
- Creates images/ folder + gt.csv
- gt.csv has 3 columns: (image_name, fen, view)

**Use it**:
```bash
# Convert dataset
python create_compliant_dataset.py --input Data --output compliant_dataset

# Verify it's correct
python create_compliant_dataset.py --output compliant_dataset --verify
```

### 4. **Created checkpoints/** - Model Storage
**Location**: `checkpoints/` folder

**What you need to do**:
1. After training in Colab, download models
2. Place best model here: `checkpoints/best_model.pth`
3. See `checkpoints/README.md` for detailed instructions

### 5. **Updated README.md** - Complete Documentation
**Location**: `README.md`

**Now includes**:
- Complete setup instructions (git clone → running)
- Training instructions for Google Colab
- Evaluation API documentation
- Web app instructions
- Dataset format specification
- Troubleshooting guide

### 6. **Created SUBMISSION_CHECKLIST.md**
**Location**: `SUBMISSION_CHECKLIST.md`

**Use it to**:
- Track what you need to submit
- Ensure nothing is forgotten
- Verify everything before deadline

---

## 🎓 Class Encodings Explained

### Official Spec Encoding (What evaluate.py Returns)
```python
0: White Pawn    | 6: Black Pawn
1: White Rook    | 7: Black Rook
2: White Knight  | 8: Black Knight
3: White Bishop  | 9: Black Bishop
4: White Queen   | 10: Black Queen
5: White King    | 11: Black King
12: Empty Square
13: OOD/Unknown
```

### Internal Model Encoding (What Your Model Outputs)
```python
0: Empty
1: P (White Pawn)    | 7: p (Black Pawn)
2: N (White Knight)  | 8: n (Black Knight)
3: B (White Bishop)  | 9: b (Black Bishop)
4: R (White Rook)    | 10: r (Black Rook)
5: Q (White Queen)   | 11: q (Black Queen)
6: K (White King)    | 12: k (Black King)
```

**Important**: `evaluate.py` handles the conversion automatically! Your model still uses internal encoding (0=empty), but `predict_board()` returns spec encoding (12=empty).

---

## 🚀 Workflow Summary

### For Training (Google Colab)

1. **Upload changed.ipynb to Colab**

2. **Create code.zip**:
   ```bash
   zip -r code.zip src/ dataset_tools/
   ```

3. **Run cells 1-9 in notebook**:
   - Cell 1-5: Setup
   - Cell 6: Prepare 7-fold splits
   - Cell 7: Train (2-3 hours)
   - Cell 8: Visualize results
   - Cell 9: Save to Google Drive

4. **Download models**:
   ```python
   # In Colab
   from google.colab import files
   files.download('dataset_out/best_model_fold_1.pth')
   ```

5. **Place locally**:
   - Copy to `checkpoints/best_model.pth`

### For Evaluation (Local Machine)

1. **Ensure model is in checkpoints/**:
   ```
   checkpoints/
   └── best_model.pth
   ```

2. **Test evaluation API**:
   ```bash
   python evaluate.py --image test_board.jpg
   ```

3. **Should see**:
   ```
   ✓ Model loaded from checkpoints/best_model.pth
   Validation accuracy: XX.XX%
   ```

### For Dataset Conversion

1. **Extract your data**:
   ```powershell
   Expand-Archive -Path all_games_data.zip -DestinationPath .
   ```

2. **Convert format**:
   ```bash
   python create_compliant_dataset.py --input Data --output compliant_dataset
   ```

3. **Verify**:
   ```bash
   python create_compliant_dataset.py --output compliant_dataset --verify
   ```

4. **Upload to drive**:
   - Upload `compliant_dataset/` to Google Drive
   - Share link with "Anyone with link can view"
   - Include link in your report

### For Web Visualization

1. **Start server**:
   ```bash
   python app.py
   ```

2. **Open browser**:
   - Visit http://localhost:5000

3. **Upload board image**:
   - Drag-drop or click to upload
   - See prediction + FEN + confidence

4. **Use for demos**:
   - Show in presentation
   - Include screenshots in report
   - Record video for webpage

---

## 📂 File Organization

### What You Have Now

```
chessboard-square-classifier/
├── evaluate.py                 # ⭐ OFFICIAL API (new)
├── app.py                      # Web app (updated with comments)
├── create_compliant_dataset.py # Dataset converter (new)
├── changed.ipynb               # Colab training notebook
├── README.md                   # Complete docs (updated)
├── SUBMISSION_CHECKLIST.md     # Submission guide (new)
├── requirements.txt            # All dependencies
│
├── checkpoints/                # For trained models (new)
│   ├── README.md              # How to get models from Colab
│   └── .gitignore             # Don't commit large .pth files
│
├── src/                        # Core code (unchanged)
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── dataset.py
│
├── dataset_tools/              # Utilities (unchanged)
│   ├── make_dataset.py
│   ├── extract_squares.py
│   ├── fen_utils.py
│   └── eval.py
│
├── templates/                  # Web app HTML (unchanged)
├── static/                     # Web app CSS/JS (unchanged)
│
├── Data/                       # Your training data
│   └── game*_per_frame/
│
└── compliant_dataset/          # Will be generated
    ├── images/
    └── gt.csv
```

---

## ✅ What Works Right Now

- ✅ Web app for visualization (http://localhost:5000)
- ✅ Training notebook for Google Colab
- ✅ Dataset conversion to compliant format
- ✅ Official evaluation API (needs trained model)
- ✅ Complete documentation in README
- ✅ Submission checklist

---

## ⚠️ What You Still Need to Do

1. **Train the model** in Google Colab (2-3 hours)
2. **Download trained model** from Colab to `checkpoints/best_model.pth`
3. **Convert dataset** to compliant format
4. **Upload dataset** to Google Drive (get share link)
5. **Upload trained models** to Google Drive (get share link)
6. **Write final report** (up to 20 pages, include ablation study)
7. **Prepare presentation** (7-10 minutes)
8. **(Optional) Create project webpage** (GitHub Pages)

---

## 🔧 Common Commands

```bash
# Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac
pip install -r requirements.txt

# Extract data
Expand-Archive -Path all_games_data.zip -DestinationPath .

# Convert dataset
python create_compliant_dataset.py --input Data --output compliant_dataset

# Test evaluation API
python evaluate.py --image test.jpg --show-tensor

# Run web app
python app.py

# Create code.zip for Colab
zip -r code.zip src/ dataset_tools/
```

---

## 📞 Quick Troubleshooting

### "No module named 'X'"
```bash
pip install -r requirements.txt
```

### "Checkpoint not found"
- Put trained model at: `checkpoints/best_model.pth`
- See: `checkpoints/README.md`

### "Board detection failed"
- Image must show clear chessboard
- Try different image
- Check lighting/angle

### Web app won't start
```bash
# Make sure you're in the right directory
cd chessboard-square-classifier
python app.py
```

### "Data/ folder not found"
```bash
# Extract first
Expand-Archive -Path all_games_data.zip -DestinationPath .
```

---

## 📊 Expected Results

After training in Colab (7-fold cross-validation):
- **Mean test accuracy**: 90-95% (depends on data quality)
- **Training time**: 2-3 hours with GPU
- **Model size**: ~100 MB per fold
- **Best fold**: Use for `checkpoints/best_model.pth`

---

## 🎯 Priorities for Next Steps

**This Week (by Jan 15)**:
1. ✅ Train model in Colab
2. ✅ Download and place trained model
3. ✅ Test evaluation API works
4. ✅ Convert dataset to compliant format

**Next Week (by Jan 20)**:
5. ✅ Write final report
6. ✅ Prepare presentation slides
7. ✅ Practice presentation

**Final Week (by Jan 24)**:
8. ✅ Present project (Jan 20-21)
9. ✅ Final submission

---

## 💡 Tips

- **Don't forget ablation study** in report (it's required!)
- **Test everything in fresh environment** before submitting
- **Include visual examples** in presentation (it's computer vision!)
- **Keep presentation under 10 minutes** (practice timing)
- **Upload datasets to drive early** (large files take time)
- **Document what didn't work** (it's viewed positively!)

---

**You're all set! Everything is adapted to course requirements.** 🎓
