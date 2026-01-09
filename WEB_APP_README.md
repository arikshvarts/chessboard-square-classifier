# Chess Board Position Classifier - Web App

## Quick Start Guide

### Prerequisites
- Python 3.8+
- pip package manager
- A trained model checkpoint (optional - app works in demo mode without one)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import flask; import torch; print('✓ All dependencies installed!')"
```

### Running the Web App

1. **Start the server:**
```bash
python app.py
```

2. **Open your browser:**
Navigate to: `http://localhost:5000`

3. **Use the app:**
 - Click "Choose File" or drag & drop a chess board image
 - Wait for the AI to analyze the position
 - View the predicted board state, FEN notation, and confidence scores
 - Download results or start a new analysis

### 🎨 Features

- **Beautiful Interface**: Modern, responsive design with smooth animations
- **Drag & Drop**: Easy image upload with drag-and-drop support
- **Visual Board**: Interactive chess board showing predicted pieces
- **FEN Notation**: Standard chess notation for importing into other software
- **Confidence Analysis**: See how confident the model is for each square
- **Export Results**: Download predictions as JSON

### Project Structure

```
chessboard-square-classifier/
├── app.py # Flask web application
├── templates/
│ └── index.html # Main web interface
├── static/
│ ├── css/
│ │ └── style.css # Styles and animations
│ └── js/
│ └── app.js # Frontend logic
├── src/ # Model and training code
│ ├── model.py
│ ├── train.py
│ ├── predict.py
│ └── dataset.py
├── dataset_tools/ # Data processing utilities
└── checkpoints/ # Trained model weights (create this folder)
```

### 🔧 Configuration

**Model Checkpoint:**
- Place your trained model checkpoint (`.pth` file) in the `checkpoints/` folder
- The app will automatically load the latest checkpoint
- Without a checkpoint, the app runs in demo mode with a pre-trained ResNet50

**Class Mapping:**
- Default classes are loaded from `dataset_out/classes.json`
- If not found, uses default 13-class mapping (empty + 12 chess pieces)

### 💡 Tips for Best Results

1. **Image Quality:**
 - Use well-lit, clear images
 - Ensure the entire board is visible
 - Avoid shadows or glare

2. **Board Detection:**
 - Square or slightly angled views work best
 - Standard chess board with clear square boundaries
 - Contrasting light/dark squares

3. **Piece Recognition:**
 - Clear piece designs (not too stylized)
 - Good contrast between pieces and board
 - Pieces centered in their squares

### Project Context

This is a Deep Learning project for automatic chess board position recognition:
- **Course**: Intro to Deep Learning (Fall 2025)
- **Team**: Ariel Shvarts & Nikol Koifman
- **Goal**: Classify each square of a chess board and reconstruct the position

### Model Details

- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Classes**: 13 (empty + 6 white pieces + 6 black pieces)
- **Input**: 224x224 RGB images
- **Output**: Square-by-square classification + confidence scores

### 🐛 Troubleshooting

**Port already in use:**
```bash
# Change port in app.py, line: app.run(debug=True, host='0.0.0.0', port=5000)
# Or kill the process using port 5000
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Model not loading:**
- Ensure checkpoint file is in `checkpoints/` folder
- Check file extension is `.pth`
- Verify checkpoint contains 'model_state_dict' key

**Board not detected:**
- Try a clearer image
- Ensure the full board is visible
- Check that the image shows a standard 8x8 chess board

### Advanced Usage

**Custom Model:**
```python
# In app.py, modify initialize_model() to load your custom architecture
MODEL = YourCustomModel(num_classes=13)
```

**API Endpoint:**
```bash
# Use the /predict endpoint programmatically
curl -X POST -F "file=@board.jpg" http://localhost:5000/predict
```

**Health Check:**
```bash
curl http://localhost:5000/health
```

### Future Enhancements

- [ ] Real-time webcam analysis
- [ ] Move suggestion based on detected position
- [ ] Multiple board styles support
- [ ] Batch processing for multiple images
- [ ] Export to popular chess formats (PGN)

### 📄 License

Educational project for academic purposes.

---

**Enjoy using the Chess Board Position Classifier! **

For questions or issues, please contact the development team.
