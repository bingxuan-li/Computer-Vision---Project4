"""
Inference script for GeoGuessr StreetView model.
Generates predictions and creates submission CSV file.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import argparse
from tqdm import tqdm

from dataset import GeoGuessrDataset
from model import GeoGuessrModel


def predict(model, dataloader, device, num_top_states=5):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained GeoGuessr model
        dataloader: DataLoader for test data
        device: Device to run inference on
        num_top_states: Number of top state predictions to return (1-5)
    
    Returns:
        predictions: List of dictionaries with predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch['images'].to(device)
            sample_ids = batch['sample_id']
            
            # Forward pass
            state_logits, gps_coords = model(images)
            
            # Get top-k state predictions
            probabilities = torch.softmax(state_logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=num_top_states, dim=1)
            
            # Process each sample in batch
            for i in range(len(sample_ids)):
                sample_id = sample_ids[i]
                
                # State predictions (1-indexed)
                state_preds = {}
                for j in range(num_top_states):
                    state_preds[f'state_idx_{j+1}'] = top_indices[i, j].item() + 1  # 1-indexed
                
                # GPS coordinates
                lat = gps_coords[i, 0].item()
                lon = gps_coords[i, 1].item()
                
                pred_dict = {
                    'sample_id': int(sample_id),
                    'latitude': lat,
                    'longitude': lon,
                    **state_preds
                }
                
                predictions.append(pred_dict)
    
    return predictions


def generate_submission_csv(predictions, output_path, data_dir):
    """
    Generate submission CSV file with required format.
    
    Required columns:
    - sample_id: Test ID
    - image_north, image_east, image_south, image_west: Filenames
    - state_idx_1: Required state prediction
    - state_idx_2 to state_idx_5: Optional state predictions
    - latitude, longitude: GPS coordinates
    """
    rows = []
    
    for pred in predictions:
        sample_id = pred['sample_id']
        
        # Format sample ID for filenames (assuming 6-digit format)
        sample_id_str = str(sample_id).zfill(6)
        
        # Create row with all required and optional fields
        row = {
            'sample_id': sample_id,
            'image_north': f'img_{sample_id_str}_north.jpg',
            'image_east': f'img_{sample_id_str}_east.jpg',
            'image_south': f'img_{sample_id_str}_south.jpg',
            'image_west': f'img_{sample_id_str}_west.jpg',
            'state_idx_1': pred.get('state_idx_1'),
            'state_idx_2': pred.get('state_idx_2', ''),
            'state_idx_3': pred.get('state_idx_3', ''),
            'state_idx_4': pred.get('state_idx_4', ''),
            'state_idx_5': pred.get('state_idx_5', ''),
            'latitude': pred['latitude'],
            'longitude': pred['longitude']
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Define column order
    columns = [
        'sample_id',
        'image_north', 'image_east', 'image_south', 'image_west',
        'state_idx_1', 'state_idx_2', 'state_idx_3', 'state_idx_4', 'state_idx_5',
        'latitude', 'longitude'
    ]
    
    df = df[columns]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f'Submission file saved to: {output_path}')
    print(f'Total samples: {len(df)}')
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for GeoGuessr StreetView')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-csv', type=str, default='submission.csv',
                        help='Path to output submission CSV file')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num-states', type=int, default=50,
                        help='Number of states in the model')
    parser.add_argument('--num-top-states', type=int, default=5,
                        help='Number of top state predictions to include (1-5)')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test dataset
    test_dataset = GeoGuessrDataset(
        data_dir=args.test_dir,
        csv_file=None,
        transform=transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f'Found {len(test_dataset)} test samples')
    
    # Load model
    model = GeoGuessrModel(num_states=args.num_states, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f'Loaded model from {args.model_path}')
    if 'epoch' in checkpoint:
        print(f'Model trained for {checkpoint["epoch"]+1} epochs')
    if 'val_accuracy' in checkpoint:
        print(f'Validation accuracy: {checkpoint["val_accuracy"]:.2f}%')
    
    # Generate predictions
    predictions = predict(model, test_loader, device, num_top_states=args.num_top_states)
    
    # Generate submission CSV
    submission_df = generate_submission_csv(predictions, args.output_csv, args.test_dir)
    
    # Display sample predictions
    print('\nSample predictions:')
    print(submission_df.head())


if __name__ == '__main__':
    main()
