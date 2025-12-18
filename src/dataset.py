"""
Dataset loader for GeoGuessr StreetView images.
Handles loading of 4-direction images (north, east, south, west) per sample.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


class GeoGuessrDataset(Dataset):
    """Dataset for loading GeoGuessr StreetView images with 4 directions."""
    
    def __init__(self, data_dir, csv_file=None, transform=None, is_test=False):
        """
        Args:
            data_dir (str): Directory containing the images
            csv_file (str): Path to CSV file with labels (for training)
            transform: Optional transform to apply to images
            is_test (bool): Whether this is test data (no labels)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        
        # Find all unique sample IDs
        self.samples = self._find_samples()
        
        # Load labels if provided
        if csv_file and not is_test:
            self.labels_df = pd.read_csv(csv_file)
        else:
            self.labels_df = None
    
    def _find_samples(self):
        """Find all unique sample IDs from the image filenames."""
        samples = set()
        
        if not os.path.exists(self.data_dir):
            return []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jpg'):
                # Extract sample ID from filename (img_XXXXXX_direction.jpg)
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 2:
                    sample_id = parts[1]
                    samples.add(sample_id)
        
        return sorted(list(samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load all 4 direction images
        directions = ['north', 'east', 'south', 'west']
        images = []
        
        for direction in directions:
            img_path = os.path.join(self.data_dir, f'img_{sample_id}_{direction}.jpg')
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                # Create a blank image if file doesn't exist
                img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
        
        # Stack images along channel dimension
        images = torch.stack(images)
        
        # Get labels if available
        if self.labels_df is not None and not self.is_test:
            row = self.labels_df[self.labels_df['sample_id'] == int(sample_id)]
            if len(row) > 0:
                # Convert state_idx from 1-based (CSV) to 0-based (for PyTorch CrossEntropyLoss)
                state_idx = torch.tensor(row.iloc[0]['state_idx_1'] - 1, dtype=torch.long)
                latitude = torch.tensor(row.iloc[0]['latitude'], dtype=torch.float32)
                longitude = torch.tensor(row.iloc[0]['longitude'], dtype=torch.float32)
                
                return {
                    'images': images,
                    'sample_id': sample_id,
                    'state_idx': state_idx,
                    'latitude': latitude,
                    'longitude': longitude
                }
        
        # Return just images for test data
        return {
            'images': images,
            'sample_id': sample_id
        }
