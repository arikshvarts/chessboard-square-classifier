# Checkpoints Directory

Place your trained model weights here after training in Google Colab.

## Folder Structure

```
checkpoints/
├── best_model.pth          # Main model for evaluation API
├── fold_1/
│   └── best_model.pth      # Fold 1 model
├── fold_2/
│   └── best_model.pth      # Fold 2 model
└── ...
```

## How to Get Models from Google Colab

After training completes in Colab (using `changed.ipynb`), download the models:

### Option 1: Download from Colab to Local

```python
# In Colab, after training completes:
from google.colab import files

# Download best overall model
files.download('dataset_out/best_model_fold_1.pth')

# Or download all folds:
!zip -r all_models.zip dataset_out/*.pth
files.download('all_models.zip')
```

Then place on your local machine:
1. Extract `all_models.zip` (if you downloaded zip)
2. Copy `best_model_fold_1.pth` to `checkpoints/best_model.pth`
3. (Optional) Copy fold models to `checkpoints/fold_*/` directories

### Option 2: Save to Google Drive (from Colab)

Cell 9 in the notebook already saves to Drive:
```python
!cp -r /content/dataset_out/ /content/drive/MyDrive/chess_models/
```

Then download from Google Drive to your computer.

### Option 3: Direct Copy from Drive

If you uploaded models to Google Drive:
1. Download from Drive folder: `MyDrive/chess_models/`
2. Copy `best_model_fold_X.pth` files to this `checkpoints/` folder
3. Rename the best one to `best_model.pth`

## File Naming

The evaluation API (`evaluate.py`) looks for models in this order:
1. `checkpoints/best_model.pth` (highest priority - use your best fold)
2. `checkpoints/fold_1/best_model.pth`
3. `dataset_out/best_model_fold_1.pth`

**Recommendation:** Copy your best-performing fold to `checkpoints/best_model.pth`

## Model File Format

Each `.pth` file should contain:
```python
{
    'model_state_dict': ...,  # Required
    'optimizer_state_dict': ...,
    'epoch': ...,
    'val_acc': ...,  # Recommended (for display)
    'fold': ...
}
```

## Checking Model Loading

Test if your model loads correctly:

```bash
python evaluate.py --image test_image.jpg
```

You should see:
```
✓ Model loaded from checkpoints/best_model.pth
  Validation accuracy: 95.23%
```

## gitignore

Model files are in `.gitignore` (they're too large for git).  
Upload them separately or provide download links in your submission.

## For Submission

In your final report, provide:
- Download link for trained models (Google Drive, Dropbox, etc.)
- Instructions for evaluators to place models in this folder
- Validation accuracy for each fold

Example:
```
Trained models: https://drive.google.com/...
Place best_model.pth in checkpoints/ folder before running evaluation.
```
