"""
Model architecture for GeoGuessr StreetView prediction.
Multi-task learning: state classification + GPS coordinate regression.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GeoGuessrModel(nn.Module):
    """
    Multi-task model for predicting:
    1. State classification (50 US states)
    2. GPS coordinates (latitude, longitude)
    """
    
    def __init__(self, num_states=50, pretrained=True):
        """
        Args:
            num_states (int): Number of states to classify
            pretrained (bool): Whether to use pretrained ResNet weights
        """
        super(GeoGuessrModel, self).__init__()
        
        # Base feature extractor (ResNet-50)
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet-50
        feature_dim = 2048
        
        # Process 4 images (north, east, south, west)
        # We'll process each separately and combine features
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # State classification head
        self.state_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_states)
        )
        
        # GPS coordinate regression head
        self.gps_regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # latitude and longitude
        )
    
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 4, 3, H, W)
                    4 directions: north, east, south, west
        
        Returns:
            state_logits: Tensor of shape (batch_size, num_states)
            gps_coords: Tensor of shape (batch_size, 2)
        """
        batch_size = images.size(0)
        num_directions = images.size(1)
        
        # Extract features from each direction
        features = []
        for i in range(num_directions):
            # Process each direction image
            img = images[:, i, :, :, :]  # (batch_size, 3, H, W)
            feat = self.feature_extractor(img)  # (batch_size, 2048, 1, 1)
            feat = feat.view(batch_size, -1)  # (batch_size, 2048)
            features.append(feat)
        
        # Concatenate features from all directions
        combined_features = torch.cat(features, dim=1)  # (batch_size, 2048*4)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)  # (batch_size, 1024)
        
        # Predict state
        state_logits = self.state_classifier(fused_features)
        
        # Predict GPS coordinates
        gps_coords = self.gps_regressor(fused_features)
        
        return state_logits, gps_coords
