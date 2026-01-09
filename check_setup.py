"""
Quick test script to verify the web app setup
Run this before starting the web app
"""

import sys
import os

print("=" * 60)
print("Chess Board Classifier - System Check")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check required packages
required_packages = [
    'torch', 'torchvision', 'PIL', 'flask', 
    'numpy', 'pandas', 'cv2'
]

print("\nChecking required packages:")
missing = []
for package in required_packages:
    try:
        if package == 'PIL':
            __import__('PIL')
        elif package == 'cv2':
            __import__('cv2')
        else:
            __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        missing.append(package)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("Run: pip install -r requirements.txt")
else:
    print("\n✓ All required packages installed!")

# Check CUDA availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  CUDA not available - using CPU")
        print("  (This is fine for inference, but training will be slow)")
except:
    pass

# Check folder structure
print("\nChecking folder structure:")
folders = ['src', 'dataset_tools', 'templates', 'static', 'static/css', 'static/js']
for folder in folders:
    if os.path.exists(folder):
        print(f"  ✓ {folder}/")
    else:
        print(f"  ✗ {folder}/ - MISSING")

# Check key files
print("\nChecking key files:")
files = [
    'app.py',
    'src/model.py',
    'dataset_tools/extract_squares.py',
    'dataset_tools/fen_utils.py',
    'templates/index.html',
    'static/css/style.css'
]
for file in files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - MISSING")

# Check for trained model
print("\nChecking for trained models:")
if os.path.exists('checkpoints'):
    models = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if models:
        print(f"  ✓ Found {len(models)} model(s) in checkpoints/")
        for model in models:
            print(f"    - {model}")
    else:
        print("  ⚠️  No .pth models in checkpoints/")
else:
    print("  ⚠️  No checkpoints/ folder")

if os.path.exists('dataset_out'):
    models = [f for f in os.listdir('dataset_out') if f.endswith('.pth')]
    if models:
        print(f"  ✓ Found {len(models)} model(s) in dataset_out/")
        for model in models[:3]:  # Show first 3
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    else:
        print("  ⚠️  No .pth models in dataset_out/")
else:
    print("  ⚠️  No dataset_out/ folder")

if not os.path.exists('checkpoints') and not os.path.exists('dataset_out'):
    print("\n  ℹ️  No trained models found.")
    print("     The app will use pretrained ResNet50 (less accurate for chess).")
    print("     Train a model using the notebook to get better results!")

# Check for data
print("\nChecking for training data:")
if os.path.exists('Data'):
    games = [d for d in os.listdir('Data') if os.path.isdir(os.path.join('Data', d))]
    if games:
        print(f"  ✓ Found {len(games)} game folder(s) in Data/")
    else:
        print("  ⚠️  Data/ folder is empty")
else:
    print("  ⚠️  No Data/ folder found")
    print("     Extract all_games_data.zip to train models")

print("\n" + "=" * 60)

if not missing:
    print("✓ System check passed! Ready to run the app.")
    print("\nTo start the web app, run:")
    print("  python app.py")
    print("\nThen open your browser to:")
    print("  http://localhost:5000")
else:
    print("⚠️  Please install missing packages first:")
    print("  pip install -r requirements.txt")

print("=" * 60)
