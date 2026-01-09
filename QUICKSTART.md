# Quick Start Guide - Chess Board Classifier

## What This Project Does

This is a **Chess Board Position Classifier** that uses Deep Learning to:
- 📸 Detect chess boards in images
- Classify each square (empty or which chess piece)
- Reconstruct the full board position
- 🌐 Display results in a beautiful web interface

---

## 🎬 How to Run the Web App (EASY!)

### Step 1: Install Dependencies
Open PowerShell in this folder and run:

```powershell
pip install -r requirements.txt
```

### Step 2: Run the Web App
```powershell
python app.py
```

### Step 3: Open Your Browser
Go to: **http://localhost:5000**

### Step 4: Upload & See Magic!
1. Click "Choose File" or drag & drop a chess board image
2. Wait a few seconds for analysis
3. See the detected board position with:
 - FEN notation
 - Visual board display
 - Confidence scores
 - Interactive square view

---

## 🧪 Training the Model (Advanced)

If you want to train your own model:

### Step 1: Prepare Data
Extract your data and place in `Data/` folder with structure:
```
Data/
 game1_per_frame/
 tagged_images/
 annotations.csv
 game2_per_frame/
 ...
```

### Step 2: Open Jupyter Notebook
```powershell
jupyter notebook "chess (1).ipynb"
```

### Step 3: Run All Cells
The notebook will:
- Create 7-fold cross-validation splits
- Train ResNet50 models on each fold
- Save best models to `dataset_out/`
- Generate performance metrics

**Note:** Training requires GPU (CUDA) for reasonable speed. On CPU it will be very slow!

---

## What's Inside the Notebook?

The **chess (1).ipynb** notebook contains:

1. **Data Preparation** - Creates 7-fold CV splits from your game data
2. **Model Architecture** - ResNet50 with 13 output classes
3. **Training Loop** - 8 epochs per fold with:
 - Adam optimizer
 - Learning rate scheduling
 - Gradient clipping
 - Early stopping
4. **Evaluation** - Test accuracy on held-out game
5. **Results** - Cross-validation summary with mean ± std accuracy

---

## 🎨 Making the Web App More Creative

### Ideas Already Implemented:
- Beautiful gradient UI with animations
- Drag & drop file upload
- Real-time square-by-square visualization
- FEN notation display
- Confidence scores per square
- Responsive design

### Ideas You Can Add:

#### 1. **Chess.js Integration** (EASY)
Add this to your HTML to display an actual chess board:
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.12.0/chess.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
```

Then render the position using the FEN!

#### 2. **Compare Before/After** (MEDIUM)
- Show original uploaded image side-by-side with detected board
- Overlay confidence heatmap on original squares

#### 3. **Move Suggestion** (HARD)
- Use Stockfish.js to suggest best move for detected position
- Show arrows on the board

#### 4. **Game History** (MEDIUM)
- Let users upload multiple frames
- Detect moves between frames
- Display full game PGN

#### 5. **Export Options** (EASY)
- Download FEN as text file
- Share position via URL
- Export as PGN format

---

## 🐛 Fixed Issues

### PyTorch Warning Fixed
- **Problem:** `verbose=True` parameter deprecated in ReduceLROnPlateau
- **Solution:** Removed verbose parameter from scheduler
- **Also Fixed:** Changed `pretrained=True` to `weights='IMAGENET1K_V1'`

---

## Testing Checklist

### Web App Testing:
- [ ] App starts without errors
- [ ] Can upload image via button
- [ ] Can drag & drop image
- [ ] Loading animation appears
- [ ] Results display correctly
- [ ] FEN notation shown
- [ ] Board visualization rendered
- [ ] Can clear and upload new image

### Notebook Testing:
- [ ] All cells run without errors
- [ ] Data loads correctly
- [ ] 7 folds created
- [ ] Models train and save
- [ ] Accuracy metrics computed
- [ ] Results saved to CSV

---

## Project Requirements (from PDFs)

### What's Completed:
1. **Data Collection** - 7 chess game videos processed
2. **Frame Extraction** - Tagged images with FEN annotations
3. **Square Extraction** - 64 squares per frame
4. **Model Architecture** - ResNet50 CNN
5. **Training** - 7-fold cross-validation
6. **Evaluation** - Per-game accuracy testing
7. **Web Interface** - Interactive demo app

### Expected Performance:
- Target: >95% square accuracy
- Cross-validation ensures generalization
- Test on unseen games (held-out fold)

---

## 🎨 How to Make Presentation More Creative

### For Demos:
1. **Live Demo**: Upload test image and show real-time detection
2. **Before/After**: Show raw image vs detected position
3. **Confidence Map**: Color-code squares by confidence
4. **Error Analysis**: Show where model struggles (similar pieces)
5. **Speed Test**: Time the full pipeline (detection + classification)

### Visual Enhancements:
1. Add piece icons instead of letters
2. Animate piece placement
3. Show detection bounding box on original image
4. Add sound effects for successful detection
5. Theme selector (light/dark/wood board)

### Interactive Features:
1. Let users click squares to see top-3 predictions
2. Show attention maps (where model looks)
3. Compare human annotation vs model prediction
4. Real-time webcam feed analysis

---

## 💡 Quick Improvements You Can Make NOW

### 1. Add Sample Images (5 min)
```python
# In app.py, add demo images endpoint
@app.route('/demo/<int:demo_id>')
def demo_image(demo_id):
 # Load pre-selected demo images
 pass
```

### 2. Add Statistics Panel (10 min)
Show on results page:
- Total pieces detected
- Material count (pawns, bishops, etc.)
- Who's winning (material advantage)

### 3. Better Error Messages (5 min)
- "No board detected? Try better lighting"
- "Board too small? Get closer"
- "Multiple boards? Crop to one"

### 4. Add Loading Tips (2 min)
While analyzing, show random chess facts!

---

## 🚨 Common Issues & Solutions

### Issue: "No module named 'torch'"
**Solution:** `pip install torch torchvision`

### Issue: "Could not detect chess board"
**Solution:**
- Make sure image shows full 8x8 board
- Check image is not rotated
- Ensure good contrast between squares

### Issue: "Model not loaded"
**Solution:** Train model first using notebook, or app will use pretrained ResNet (less accurate)

### Issue: Port 5000 already in use
**Solution:** Change port in app.py: `app.run(port=5001)`

---

## Next Steps

1. **Test the Web App** - Upload sample chess board images
2. **Review Notebook** - Understand the training process
3. **Train Your Model** - Use your data to get better accuracy
4. **Customize UI** - Make it your own style
5. **Add Features** - Implement the creative ideas above

---

## Presenting This Project

### Demo Flow:
1. **Introduction** (1 min)
 - "Chess position recognition using Deep Learning"
2. **Show Web App** (2 min)
 - Upload image
 - Show detection process
 - Explain FEN output
3. **Explain Architecture** (2 min)
 - Show ResNet50 diagram
 - Explain square extraction
 - Discuss 13 classes
4. **Show Results** (2 min)
 - Display accuracy metrics
 - Show confusion matrix
 - Discuss improvements
5. **Live Q&A** (3 min)
 - Take questions
 - Demo edge cases

### Key Points to Mention:
- 7-fold cross-validation for robust evaluation
- Transfer learning from ImageNet
- Real-world data from actual games
- Handles various lighting/angles
- Production-ready web interface

---

## 🎉 You're Ready!

Just run: `python app.py` and start uploading chess board images!

Need help? Check:
- `WEB_APP_README.md` - Detailed web app docs
- `README.md` - General project overview
- PDFs in root folder - Project requirements

** **
