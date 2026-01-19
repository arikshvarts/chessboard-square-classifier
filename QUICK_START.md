# Quick Start - Testing Your Code

## ✅ What's Ready

All code is implemented and pushed to GitHub. Your trained model exists at:
- `checkpoints/best_model.pth` (282 MB)
- `checkpoints/best_model_fold_1.pth` (282 MB)

## 🚀 Quick Test Commands

**Always use the virtual environment Python:**

```powershell
# Activate venv (if needed)
..\..\.venv\Scripts\Activate.ps1

# OR use full path
$py = "C:/Users/ariks/uni/DeepLearning/Final_miss_clone_for_web_App/.venv/Scripts/python.exe"

# Test evaluation API (using sample image)
& $py evaluate.py --image docs/assets/sample_debug_grid.png --show-tensor --save-viz

# Test demo
& $py demo.py --image docs/assets/sample_debug_grid.png

# Check if you have training data
Test-Path "Data"

# If you have Data/, generate compliant dataset
& $py create_compliant_dataset.py --input Data --output compliant_dataset

# Verify dataset format
& $py create_compliant_dataset.py --output compliant_dataset --verify
```

## 📦 What You MUST Upload to Google Drive

**REQUIRED by instructor:**

1. **Trained Model** (already have it):
   - `checkpoints/best_model.pth` (282 MB)

2. **Compliant Dataset** (need to generate if you have Data/):
   - Run: `create_compliant_dataset.py`
   - Upload: `compliant_dataset/` folder (with images/ and gt.csv)

3. **Original Data** (if different from compliant):
   - Your `Data/` folder (if you used it for training)

## 📊 Submission Checklist (Deadline: Jan 24, 2026)

### GitHub (Already Done ✓):
- ✅ Code pushed
- ✅ README complete
- ✅ requirements.txt present
- ✅ evaluate.py with `predict_board()`
- ✅ demo.py exists

### Google Drive (TODO):
- ⏳ Upload `best_model.pth` (282 MB)
- ⏳ Upload `compliant_dataset/` (if you have Data/)
- ⏳ Get shareable link

### Documents (TODO):
- ⏳ Write final report PDF (max 25 pages)
- ⏳ Create presentation (7-10 min)
- ⏳ **Include ablation study in report** (REQUIRED)

## ⚠️ Important Notes

1. **Virtual Environment**: Always use the venv Python, not system Python
2. **Model File**: best_model.pth exists (copied from best_model_fold_1.pth)
3. **OOD Visualization**: Code saves red X marks to `./results/` folder
4. **Coordinate Convention**: output[0,0] = top-left of IMAGE (correct!)

## 🎯 Next Immediate Actions

1. **Test your code** (if sample image works)
2. **Generate compliant dataset** (if you have Data/ folder)
3. **Upload to Google Drive** (model + dataset)
4. **Write report** (include ablation study!)
5. **Prepare presentation** (7-10 minutes)

**5 days left until deadline!** ⏰
