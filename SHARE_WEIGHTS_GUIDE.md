# Quick Guide: Share Model Weights with Your Friend

## Current Status

✅ **Code is in git** - Your friend can clone it
✅ **Model weights exist locally** - `checkpoints/best_model_fold_1.pth` (270 MB)
❌ **Weights NOT in git** - Too large, excluded by .gitignore

## What Your Friend Needs

Your friend needs the file: `checkpoints/best_model_fold_1.pth` (270 MB)

---

## Step 1: Upload to Google Drive (Easiest)

### Using Google Drive Web Interface:
1. Go to https://drive.google.com
2. Click "New" → "File upload"
3. Navigate to: `checkpoints/best_model_fold_1.pth`
4. Wait for upload to complete
5. Right-click the file → "Share" → "Anyone with the link"
6. Copy the link
7. Send link to your friend

### Or using Google Drive Desktop App (if installed):
1. Copy `best_model_fold_1.pth` to your Google Drive folder
2. Wait for sync
3. Share as above

---

## Step 2: Send Instructions to Your Friend

**Message to send:**

```
Hey! To run the chess web app:

1. Clone the repo from GitHub
2. Download the trained model weights from this link:
   [YOUR GOOGLE DRIVE LINK HERE]
3. Place the downloaded file in: checkpoints/best_model_fold_1.pth
4. Follow SETUP_FOR_FRIEND.md in the repo

The model is 270 MB and has 98.77% validation accuracy.
Let me know if you have issues!
```

---

## Alternative Options

### Option A: WeTransfer (No Account Needed)
1. Go to https://wetransfer.com
2. Click "Add your files"
3. Select `best_model_fold_1.pth`
4. Enter friend's email
5. Send - they'll get download link

### Option B: Dropbox
1. Go to https://dropbox.com
2. Upload file
3. Get shareable link

### Option C: Direct Share (Physical)
- Copy to USB drive
- Or use local network file sharing

---

## Verification Script for Your Friend

After your friend downloads, they should run:

```bash
# Windows PowerShell
Test-Path checkpoints/best_model_fold_1.pth
(Get-Item checkpoints/best_model_fold_1.pth).Length / 1MB

# Should output:
# True
# 269.77 (size in MB)
```

Then start the app:
```bash
python app.py
```

Expected output:
```
Using device: cpu
Loading model from: checkpoints/best_model_fold_1.pth
Detected Colab checkpoint format (direct ResNet50)
Model loaded successfully! Val Acc: 98.77
 * Running on http://127.0.0.1:5000
```

---

## Files Already in Git (Friend Can Clone)

✅ app.py - Web application
✅ src/ - Model code
✅ dataset_tools/ - Data processing
✅ requirements.txt - Dependencies
✅ README.md - Full documentation
✅ SETUP_FOR_FRIEND.md - Setup guide
✅ checkpoints/.gitignore - Folder structure
✅ All other code files

## Files NOT in Git (Need Separate Sharing)

❌ checkpoints/best_model_fold_1.pth (270 MB) - **THIS ONE!**
❌ Data/ folder (training data)
❌ dataset_out/ folder (training outputs)

---

## Quick Upload Command (if you have gdown or similar)

If your friend is comfortable with command line and you have a Google Drive link:

```bash
# Your friend can run this after cloning:
pip install gdown
gdown [YOUR_GOOGLE_DRIVE_FILE_ID] -O checkpoints/best_model_fold_1.pth
```

To get FILE_ID from Google Drive link:
- Link format: `https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing`
- Copy the FILE_ID part

---

## Summary

1. ✅ Git repo has all the code
2. ❌ Git repo does NOT have model weights (too large)
3. 📤 You need to upload `best_model_fold_1.pth` to cloud storage
4. 📧 Send your friend the download link + SETUP_FOR_FRIEND.md instructions
5. ✅ Friend downloads weights, places in checkpoints/, runs app

**Total time: ~5 minutes upload, ~2 minutes download for your friend**
