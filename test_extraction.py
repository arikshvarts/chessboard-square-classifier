"""
Test script to verify square extraction works correctly.
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
import json

# Import the extraction function
from dataset_tools.extract_squares import extract_64_square_crops

print("✓ Successfully imported extract_64_square_crops\n")


def find_data_files():
    """Locate the manifest and classes files."""
    if os.path.exists('dataset_out/dataset_manifest.csv'):
        return 'dataset_out/dataset_manifest.csv', 'dataset_out/classes.json'
    elif os.path.exists('data/dataset_manifest.csv'):
        return 'data/dataset_manifest.csv', 'data/classes.json'
    else:
        print(" Could not find dataset files")
        sys.exit(1)


def extract_single_square(img, square_idx):
    """Extract a single square by getting all 64 and selecting one."""
    all_squares = extract_64_square_crops(img)
    return all_squares[square_idx]


def test_single_square_extraction():
    """Test extracting a single square from an image."""
    print("=" * 60)
    print("TEST 1: Single Square Extraction")
    print("=" * 60)
    
    manifest_path, classes_path = find_data_files()
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(manifest)} rows")
    
    # Get a sample with a piece (non-empty)
    with open(classes_path, 'r') as f:
        class_map = json.load(f)
    
    non_empty = manifest[manifest['label_id'] != 0]
    if len(non_empty) > 0:
        sample_row = non_empty.iloc[0]
    else:
        sample_row = manifest.iloc[0]
    
    print(f"\nSample row:")
    for col in sample_row.index:
        print(f"  {col}: {sample_row[col]}")
    
    img_path = sample_row['frame_path']
    print(f"\nLoading image: {img_path}")
    
    if not os.path.exists(img_path):
        print(f" Image not found: {img_path}")
        return False
    
    img = Image.open(img_path).convert('RGB')
    print(f"✓ Image loaded: {img.size}")
    
    # Extract one square
    square_idx = int(sample_row['square_idx'])
    label_id = int(sample_row['label_id'])
    piece_name = class_map[str(label_id)]
    
    print(f"\nExtracting square_idx: {square_idx}")
    print(f"Expected label: {label_id} ({piece_name})")
    
    try:
        square_img = extract_single_square(img, square_idx)
        print(f"✓ Square extracted: {square_img.size}")
    except Exception as e:
        print(f" Error extracting square: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(square_img)
    ax2.set_title(f'Square {square_idx}\nLabel: {piece_name}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_single_square.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to test_single_square.png")
    plt.close()
    
    return True


def test_all_squares_extraction():
    """Test extracting all 64 squares."""
    print("\n" + "=" * 60)
    print("TEST 2: All 64 Squares Extraction & Ordering")
    print("=" * 60)
    
    manifest_path, classes_path = find_data_files()
    
    # Load classes
    with open(classes_path, 'r') as f:
        class_map = json.load(f)
    
    print(f"Classes: {class_map}\n")
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    
    # Get first image with all 64 squares
    first_image = manifest['frame_path'].iloc[0]
    image_squares = manifest[manifest['frame_path'] == first_image].sort_values('square_idx')
    
    print(f"Loading image: {first_image}")
    print(f"Labeled squares for this image: {len(image_squares)}")
    
    if not os.path.exists(first_image):
        print(f"Image not found: {first_image}")
        return False
    
    img = Image.open(first_image).convert('RGB')
    print(f"✓ Image loaded: {img.size}")
    
    # Extract all 64 squares
    print("\nExtracting all 64 squares...")
    try:
        all_squares = extract_64_square_crops(img)
        print(f"✓ Extracted {len(all_squares)} squares")
        
        if len(all_squares) != 64:
            print(f"⚠ Warning: Expected 64 squares, got {len(all_squares)}")
        
        # Check first square size
        print(f"  Each square size: {all_squares[0].size}")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get labels for this image
    labels = {}
    for _, row in image_squares.iterrows():
        square_idx = int(row['square_idx'])
        label_id = int(row['label_id'])
        piece = class_map[str(label_id)]
        labels[square_idx] = piece
    
    # Create visualization grid
    print("\nCreating visualization grid...")
    
    square_size = 80
    canvas_width = square_size * 8
    canvas_height = square_size * 8
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 11)
        except:
            font = ImageFont.load_default()
    
    # Place squares in grid
    for square_idx in range(min(64, len(all_squares))):
        row = square_idx // 8
        col = square_idx % 8
        
        x = col * square_size
        y = row * square_size
        
        # Resize and paste square
        square_resized = all_squares[square_idx].resize((square_size, square_size), Image.Resampling.LANCZOS)
        canvas.paste(square_resized, (x, y))
        
        # Draw label with background
        piece = labels.get(square_idx, '?')
        label_text = f"{square_idx}:{piece}"
        
        # Draw semi-transparent yellow background
        draw.rectangle([x, y, x + 55, y + 16], fill='yellow', outline='black')
        draw.text((x + 3, y + 2), label_text, fill='black', font=font)
    
    canvas.save('test_all_squares_grid.png')
    print("✓ Saved to test_all_squares_grid.png")
    
    # Also create a matplotlib figure
    fig = plt.figure(figsize=(14, 14))
    plt.imshow(canvas)
    plt.title('All 64 Squares - Verify Orientation!\n(Square 0 = top-left, Square 63 = bottom-right)', 
              fontsize=14, pad=20)
    plt.axis('off')
    
    # Add file rank labels
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    plt.tight_layout()
    plt.savefig('test_all_squares_grid_display.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved to test_all_squares_grid_display.png")
    
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKLIST:")
    print("=" * 60)
    print("Open test_all_squares_grid.png and verify:")
    print("1. Square 0 (top-left) should be rank 8, file a")
    print("2. Square 7 (top-right) should be rank 8, file h") 
    print("3. Square 56 (bottom-left) should be rank 1, file a")
    print("4. Square 63 (bottom-right) should be rank 1, file h")
    print("5. Piece labels should match what you see in the image")
    print("=" * 60)
    
    return True


def test_fen_reconstruction():
    """Test FEN reconstruction from labels."""
    print("\n" + "=" * 60)
    print("TEST 3: FEN Reconstruction")
    print("=" * 60)
    
    manifest_path, classes_path = find_data_files()
    
    # Load classes
    with open(classes_path, 'r') as f:
        class_map = json.load(f)
    id_to_piece = {int(k): v for k, v in class_map.items()}
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    
    # Get one complete board
    first_image = manifest['frame_path'].iloc[0]
    image_squares = manifest[manifest['frame_path'] == first_image].sort_values('square_idx')
    
    print(f"Image: {first_image}")
    print(f"Reconstructing FEN from {len(image_squares)} labeled squares...\n")
    
    # Build board array [8 rows x 8 cols]
    board = [[None for _ in range(8)] for _ in range(8)]
    
    for _, row in image_squares.iterrows():
        square_idx = int(row['square_idx'])
        label_id = int(row['label_id'])
        piece = id_to_piece[label_id]
        
        # Convert square_idx to board position
        board_row = square_idx // 8
        board_col = square_idx % 8
        
        if piece != 'empty':
            board[board_row][board_col] = piece
    
    # Convert to FEN
    fen_ranks = []
    for rank_idx in range(8):
        fen_rank = ""
        empty_count = 0
        
        for file_idx in range(8):
            piece = board[rank_idx][file_idx]
            
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece
        
        if empty_count > 0:
            fen_rank += str(empty_count)
        
        fen_ranks.append(fen_rank)
    
    fen = '/'.join(fen_ranks)
    
    print(f"Reconstructed FEN:")
    print(f"  {fen}")
    
    # Try to visualize with chess library
    try:
        import chess
        import chess.svg
        
        full_fen = fen + " w KQkq - 0 1"
        chess_board = chess.Board(full_fen)
        
        print("\nBoard visualization (text):")
        print(chess_board)
        
        # Save SVG
        svg = chess.svg.board(chess_board, size=400)
        with open('test_fen_board.svg', 'w') as f:
            f.write(svg)
        print("\n✓ Saved board visualization to test_fen_board.svg")
        
    except ImportError:
        print("\n⚠ Install python-chess to visualize: pip install python-chess")
    except Exception as e:
        print(f"\n⚠ Could not create chess board: {e}")
        print("   This might indicate a FEN format issue")
    
    return True


if __name__ == '__main__':
    print("Testing Square Extraction Pipeline")
    print("Working directory:", os.getcwd())
    print()
    
    success = True
    
    try:
        if not test_single_square_extraction():
            success = False
        
        if not test_all_squares_extraction():
            success = False
        
        if not test_fen_reconstruction():
            success = False
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
            print("\nGenerated files:")
            print("  - test_single_square.png")
            print("  - test_all_squares_grid.png")
            print("  - test_all_squares_grid_display.png")
            print("  - test_fen_board.svg (if python-chess installed)")
            print("\nNext step: Review the images and proceed to create dataset.py")
        else:
            print("\n⚠ Some tests failed - review errors above")
        
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()