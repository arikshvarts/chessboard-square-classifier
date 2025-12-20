import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import argparse
from tqdm import tqdm

from dataset import ChessSquareDataset
from model import ChessSquareClassifier, count_parameters


def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(5),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_dataloaders(manifest_path, classes_path, batch_size=64, num_workers=4):
    import pandas as pd
    
    # Load manifest and split
    df = pd.read_csv(manifest_path)
    
    loaders = {}
    
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name]
        
        if len(split_df) == 0:
            print(f"Warning: No data for split '{split_name}'")
            continue
        
        # Save temporary CSV
        temp_csv = f'temp_{split_name}_manifest.csv'
        split_df.to_csv(temp_csv, index=False)
        
        # Get transforms
        transform = get_transforms(is_training=(split_name == 'train'))
        
        # Create dataset
        dataset = ChessSquareDataset(
            temp_csv,
            classes_path,
            transform=transform,
            cache_images=(split_name == 'train')  # Cache training data
        )
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == 'train')
        )
        
        loaders[split_name] = loader
        print(f"Created {split_name} loader: {len(dataset)} samples, {len(loader)} batches")
    
    return loaders


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
     
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
     
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, split_name='Val'):
  
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'{split_name}')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def train(args):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    print("\n" + "=" * 60)
    print("Creating dataloaders...")
    print("=" * 60)
    
    loaders = create_dataloaders(
        args.manifest,
        args.classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    
    model = ChessSquareClassifier(
        num_classes=13,
        pretrained=True,
        model_name=args.model
    )
    model = model.to(device)
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, loaders['val'], criterion, device, split_name='Val'
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print(f"  New best model! Val Acc: {val_acc:.2f}%")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_acc,
            os.path.join(args.output_dir, 'latest_model.pth')
        )
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    # Final test evaluation
    if 'test' in loaders:
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)
        
        test_loss, test_acc = validate(
            model, loaders['test'], criterion, device, split_name='Test'
        )
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train chess square classifier')
    
    # Data
    parser.add_argument('--manifest', type=str, default='dataset_out/dataset_manifest.csv',
                        help='Path to dataset manifest CSV')
    parser.add_argument('--classes', type=str, default='dataset_out/classes.json',
                        help='Path to classes JSON')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)