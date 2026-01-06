import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset_tools.extract_squares import extract_64_square_crops


class ChessSquareDataset(Dataset):
  
    def __init__(self, manifest_csv, classes_json, transform=None, cache_images=False):
        self.df = pd.read_csv(manifest_csv)
        
        with open(classes_json, 'r') as f:
            self.class_map = json.load(f)
        
        self.num_classes = len(self.class_map)
        self.transform = transform
        self.cache_images = cache_images
        
 
        self.square_cache = {} if cache_images else None
        
        if cache_images:
            print(f"Caching squares from {self.df['frame_path'].nunique()} unique images...")
            self._cache_all_squares()
        
        print(f"Dataset initialized: {len(self.df)} samples, {self.num_classes} classes")
    
    def _cache_all_squares(self):
        """Pre-extract and cache all squares from all unique images."""
        unique_images = self.df['frame_path'].unique()
        
        for i, img_path in enumerate(unique_images):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            img = Image.open(img_path).convert('RGB')
            all_squares = extract_64_square_crops(img)
            self.square_cache[img_path] = all_squares
            
            if (i + 1) % 20 == 0:
                print(f"  Cached {i + 1}/{len(unique_images)} images...")
        
        print(f"✓ Cached {len(self.square_cache)} images with {len(self.square_cache) * 64} squares")
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        square_idx = int(row['square_idx'])
        label = int(row['label_id'])
        
       
        if self.cache_images and self.square_cache is not None:
            if img_path not in self.square_cache:
                img = Image.open(img_path).convert('RGB')
                self.square_cache[img_path] = extract_64_square_crops(img)
            
            square_img = self.square_cache[img_path][square_idx]
        else:
           
            img = Image.open(img_path).convert('RGB')
            all_squares = extract_64_square_crops(img)
            square_img = all_squares[square_idx]
        
   
        if self.transform:
            square_img = self.transform(square_img)
        
        return square_img, label
    
    def get_class_name(self, label_id):
        """Convert label ID to piece name (e.g., 7 -> 'p')."""
        return self.class_map[str(label_id)]
    
    def get_label_distribution(self):
        """Print and return class distribution."""
        dist = self.df['label_id'].value_counts().sort_index()
        
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)
        for label_id, count in dist.items():
            piece_name = self.get_class_name(label_id)
            percentage = 100 * count / len(self.df)
            print(f"  {label_id:2d} ({piece_name:5s}): {count:6d} samples ({percentage:5.2f}%)")
        print("=" * 60)
        
        return dist


def create_dataloaders(manifest_path, classes_path, batch_size=64, num_workers=4, cache_train=True):
    from torchvision import transforms
    
    # Training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(5),  # Small rotation - chess pieces have orientation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation/test: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load full manifest
    df = pd.read_csv(manifest_path)
    
    loaders = {}
    
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name]
        
        if len(split_df) == 0:
            print(f"Warning: No data for split '{split_name}'")
            continue
        
        # Save temporary CSV for this split
        temp_csv = f'temp_{split_name}_manifest.csv'
        split_df.to_csv(temp_csv, index=False)
        
        # Choose transform and caching
        transform = train_transform if split_name == 'train' else val_transform
        cache = cache_train if split_name == 'train' else False
        
        # Create dataset
        dataset = ChessSquareDataset(
            temp_csv, 
            classes_path, 
            transform=transform,
            cache_images=cache
        )
        
        # Create dataloader
        shuffle = (split_name == 'train')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == 'train')  # Drop incomplete batches in training
        )
        
        loaders[split_name] = loader
        print(f"✓ Created {split_name} loader: {len(dataset)} samples, {len(loader)} batches")
    
    return loaders


if __name__ == '__main__':
    """Test the dataset loading."""
    from torchvision import transforms
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("TESTING ChessSquareDataset")
    print("=" * 60)
    
    # Define simple transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Check paths
    manifest_path = 'dataset_out/dataset_manifest.csv'
    classes_path = 'dataset_out/classes.json'
    
    if not os.path.exists(manifest_path):
        print(f"\nManifest not found: {manifest_path}")
        print("Run: python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out")
        sys.exit(1)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ChessSquareDataset(
        manifest_csv=manifest_path,
        classes_json=classes_path,
        transform=transform,
        cache_images=False  
    )
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")
    
    # Show class distribution
    dataset.get_label_distribution()
    
    # Test loading samples
    print("\n" + "=" * 60)
    print("TESTING SAMPLE LOADING")
    print("=" * 60)
    
    for i in range(5):
        img, label = dataset[i]
        print(f"Sample {i}: shape={img.shape}, label={label} ({dataset.get_class_name(label)})")
    
    # Visualize samples - one per class
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    
    # Get one sample per class
    samples_per_class = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in samples_per_class:
            samples_per_class[label] = i
        if len(samples_per_class) >= 13:  # All 13 classes
            break
    
    # Plot samples
    for ax_idx, (label, dataset_idx) in enumerate(samples_per_class.items()):
        if ax_idx >= 15:
            break
        
        img, label = dataset[dataset_idx]
        
        # Convert tensor to numpy for display [C,H,W] -> [H,W,C]
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[ax_idx].imshow(img_np)
        axes[ax_idx].set_title(
            f'Class {label}: {dataset.get_class_name(label)}', 
            fontsize=11, 
            fontweight='bold'
        )
        axes[ax_idx].axis('off')
    
    # Hide unused subplots
    for ax_idx in range(len(samples_per_class), 15):
        axes[ax_idx].axis('off')
    
    plt.suptitle('Chess Square Dataset - Sample Images (One Per Class)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to dataset_samples.png")
    plt.show()
    
    # Test DataLoader
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Get one batch
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")  # Should be [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")  # Should be [32]
    print(f"Sample labels in batch: {labels[:10].tolist()}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nDataset is ready for training!")
    print("Next step: Create train.py")