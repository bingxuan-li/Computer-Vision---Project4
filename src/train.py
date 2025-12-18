"""
Training script for GeoGuessr StreetView model.
Trains a multi-task model for state classification and GPS coordinate regression.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from dataset import GeoGuessrDataset
from model import GeoGuessrModel


def train_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['images'].to(device)
        state_idx = batch['state_idx'].to(device)
        latitude = batch['latitude'].to(device)
        longitude = batch['longitude'].to(device)
        
        # Forward pass
        state_logits, gps_coords = model(images)
        
        # Calculate losses
        cls_loss = criterion_cls(state_logits, state_idx)
        
        # GPS regression loss
        gps_target = torch.stack([latitude, longitude], dim=1)
        reg_loss = criterion_reg(gps_coords, gps_target)
        
        # Combined loss (weighted)
        loss = cls_loss + 0.1 * reg_loss  # Weight GPS loss lower
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_reg_loss


def validate(model, dataloader, criterion_cls, criterion_reg, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(device)
            state_idx = batch['state_idx'].to(device)
            latitude = batch['latitude'].to(device)
            longitude = batch['longitude'].to(device)
            
            # Forward pass
            state_logits, gps_coords = model(images)
            
            # Calculate losses
            cls_loss = criterion_cls(state_logits, state_idx)
            
            gps_target = torch.stack([latitude, longitude], dim=1)
            reg_loss = criterion_reg(gps_coords, gps_target)
            
            loss = cls_loss + 0.1 * reg_loss
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(state_logits, 1)
            total += state_idx.size(0)
            correct += (predicted == state_idx).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_cls_loss, avg_reg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train GeoGuessr StreetView model')
    parser.add_argument('--data-dir', type=str, default='data/train',
                        help='Directory containing training images')
    parser.add_argument('--csv-file', type=str, default='data/train_labels.csv',
                        help='Path to training labels CSV')
    parser.add_argument('--val-dir', type=str, default='data/val',
                        help='Directory containing validation images')
    parser.add_argument('--val-csv', type=str, default='data/val_labels.csv',
                        help='Path to validation labels CSV')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num-states', type=int, default=50,
                        help='Number of states to classify')
    parser.add_argument('--checkpoint-dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = GeoGuessrDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=transform,
        is_test=False
    )
    
    val_dataset = GeoGuessrDataset(
        data_dir=args.val_dir,
        csv_file=args.val_csv,
        transform=transform,
        is_test=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Model
    model = GeoGuessrModel(num_states=args.num_states, pretrained=True)
    model = model.to(device)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_cls_loss, train_reg_loss = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )
        print(f'Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})')
        
        # Validate
        val_loss, val_cls_loss, val_reg_loss, val_acc = validate(
            model, val_loader, criterion_cls, criterion_reg, device
        )
        print(f'Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Reg: {val_reg_loss:.4f})')
        print(f'Val Accuracy: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        
        # Save latest model
        latest_path = os.path.join(args.checkpoint_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, latest_path)
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()
