"""
Dataset Conversion Script: Raw Data -> Compliant Format

This script converts your game-based dataset to the required format:
    dataset_root/
    ├── images/
    └── gt.csv

gt.csv columns:
    1. image_name (e.g., frame_001234.jpg)
    2. FEN string (board position)
    3. View specification (white_bottom or black_bottom)

Usage:
    python create_compliant_dataset.py --input Data --output compliant_dataset

This creates a standardized dataset format for evaluation and submission.
"""

import os
import glob
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def determine_view_from_game(game_id):
    """
    Determine camera view based on game ID or metadata.
    You should adjust this based on your actual data.
    
    Returns:
        'white_bottom' or 'black_bottom'
    """
    # Default assumption: white pieces are closer to camera
    # Adjust this mapping based on your actual game recordings
    
    # Example mapping (update based on your games):
    white_bottom_games = ['game2_per_frame', 'game4_per_frame', 'game6_per_frame']
    black_bottom_games = ['game5_per_frame', 'game7_per_frame']
    
    if game_id in white_bottom_games:
        return 'white_bottom'
    elif game_id in black_bottom_games:
        return 'black_bottom'
    else:
        # Default: assume white is at bottom
        return 'white_bottom'


def convert_dataset(input_root, output_root, copy_images=True):
    """
    Convert game-based dataset to compliant format.
    
    Args:
        input_root: Path to Data/ folder with game*_per_frame subdirectories
        output_root: Path where compliant dataset will be created
        copy_images: If True, copy images. If False, create symlinks (faster but Windows needs admin)
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Create output directories
    images_dir = output_path / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DATASET CONVERSION: Raw -> Compliant Format")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Method: {'Copy images' if copy_images else 'Create symlinks'}")
    print("="*60)
    
    # Find all game directories
    game_dirs = sorted(glob.glob(str(input_path / '*_per_frame')))
    
    if not game_dirs:
        print(f"⚠ No game directories found in {input_path}")
        print("  Expected directories like: game2_per_frame, game4_per_frame, etc.")
        return
    
    print(f"\nFound {len(game_dirs)} game directories:")
    for game_dir in game_dirs:
        print(f"  - {os.path.basename(game_dir)}")
    
    # Collect all data
    all_rows = []
    total_frames = 0
    image_counter = 0
    
    for game_dir in game_dirs:
        game_id = os.path.basename(game_dir)
        print(f"\n Processing {game_id}...")
        
        # Find CSV file
        csv_files = glob.glob(os.path.join(game_dir, '*.csv'))
        if not csv_files:
            print(f"  ⚠ No CSV found in {game_dir}")
            continue
        
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        
        # Determine frame column name
        frame_col = 'from_frame' if 'from_frame' in df.columns else 'frame_id'
        
        if frame_col not in df.columns:
            print(f"  ⚠ No frame column found in {csv_path}")
            continue
        
        if 'fen' not in df.columns:
            print(f"  ⚠ No FEN column found in {csv_path}")
            continue
        
        # Determine view
        view = determine_view_from_game(game_id)
        
        # Process each frame
        tagged_images_dir = os.path.join(game_dir, 'tagged_images')
        
        if not os.path.exists(tagged_images_dir):
            print(f"  ⚠ No tagged_images/ folder in {game_dir}")
            continue
        
        game_frames = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {game_id}"):
            frame_id = int(row[frame_col])
            fen = row['fen']
            
            # Find source image
            source_image = os.path.join(tagged_images_dir, f'frame_{frame_id:06d}.jpg')
            
            if not os.path.exists(source_image):
                # Try without zero-padding
                source_image = os.path.join(tagged_images_dir, f'frame_{frame_id}.jpg')
                
            if not os.path.exists(source_image):
                continue
            
            # Create new image name (unique across all games)
            image_counter += 1
            new_image_name = f'frame_{image_counter:06d}.jpg'
            dest_image = images_dir / new_image_name
            
            # Copy or link image
            if copy_images:
                shutil.copy2(source_image, dest_image)
            else:
                try:
                    os.symlink(os.path.abspath(source_image), dest_image)
                except OSError:
                    # Symlink failed (Windows without admin), fall back to copy
                    shutil.copy2(source_image, dest_image)
            
            # Add row to dataset
            all_rows.append({
                'image_name': new_image_name,
                'fen': fen,
                'view': view,
                'source_game': game_id,
                'source_frame': frame_id
            })
            
            game_frames += 1
        
        print(f"  ✓ Processed {game_frames} frames")
        total_frames += game_frames
    
    # Create gt.csv (required 3 columns only)
    print(f"\nCreating gt.csv...")
    gt_df = pd.DataFrame(all_rows)
    
    # Save with only required columns
    required_columns = ['image_name', 'fen', 'view']
    gt_df[required_columns].to_csv(output_path / 'gt.csv', index=False)
    
    # Save extended version with metadata (for debugging)
    gt_df.to_csv(output_path / 'gt_extended.csv', index=False)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total frames: {total_frames}")
    print(f"Images in:    {images_dir}")
    print(f"Ground truth: {output_path / 'gt.csv'}")
    print("\nDataset structure:")
    print(f"  {output_path}/")
    print(f"  ├── images/          ({total_frames} images)")
    print(f"  ├── gt.csv           (required format: 3 columns)")
    print(f"  └── gt_extended.csv  (with source metadata)")
    print("="*60)
    
    # Show sample rows
    print("\nSample gt.csv rows:")
    print(gt_df[required_columns].head(3).to_string(index=False))
    
    # Statistics
    print(f"\nView distribution:")
    print(gt_df['view'].value_counts().to_string())
    
    print(f"\nSource games:")
    print(gt_df['source_game'].value_counts().to_string())
    
    print("\n✓ Dataset is now in compliant format for submission!")


def verify_dataset(dataset_root):
    """Verify that the dataset follows the required format"""
    dataset_path = Path(dataset_root)
    
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    errors = []
    warnings = []
    
    # Check images/ folder exists
    images_dir = dataset_path / 'images'
    if not images_dir.exists():
        errors.append("Missing images/ folder")
    else:
        image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        print(f"✓ images/ folder exists ({image_count} images)")
    
    # Check gt.csv exists
    gt_csv = dataset_path / 'gt.csv'
    if not gt_csv.exists():
        errors.append("Missing gt.csv file")
    else:
        print(f"✓ gt.csv exists")
        
        # Check CSV format
        df = pd.read_csv(gt_csv)
        
        if len(df.columns) < 3:
            errors.append(f"gt.csv has only {len(df.columns)} columns (need 3)")
        else:
            print(f"✓ gt.csv has {len(df.columns)} columns")
        
        expected_cols = ['image_name', 'fen', 'view']
        for col in expected_cols:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
        
        if not errors:
            print(f"✓ All required columns present: {expected_cols}")
            
            # Check each image exists
            missing_images = 0
            for img_name in df['image_name']:
                if not (images_dir / img_name).exists():
                    missing_images += 1
            
            if missing_images > 0:
                warnings.append(f"{missing_images} images referenced in CSV but not found")
            else:
                print(f"✓ All {len(df)} images exist")
            
            # Check view values
            valid_views = {'white_bottom', 'black_bottom'}
            invalid_views = set(df['view'].unique()) - valid_views
            if invalid_views:
                warnings.append(f"Non-standard view values: {invalid_views}")
            else:
                print(f"✓ View values are valid: {df['view'].unique()}")
    
    print("\n" + "="*60)
    if errors:
        print("❌ VALIDATION FAILED")
        for err in errors:
            print(f"  ❌ {err}")
    else:
        print("✅ VALIDATION PASSED")
    
    if warnings:
        print("\n⚠ Warnings:")
        for warn in warnings:
            print(f"  ⚠ {warn}")
    
    print("="*60)
    
    return len(errors) == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert dataset to compliant format for submission'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='Data',
        help='Input directory with game*_per_frame folders (default: Data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='compliant_dataset',
        help='Output directory for compliant dataset (default: compliant_dataset)'
    )
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Use symlinks instead of copying images (faster, needs admin on Windows)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing dataset format'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output)
    else:
        convert_dataset(args.input, args.output, copy_images=not args.no_copy)
        print("\nVerifying created dataset...")
        verify_dataset(args.output)
