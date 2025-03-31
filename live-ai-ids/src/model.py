#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model.py - ML model architecture for network intrusion detection

This module implements:
1. The Deep Concatenated CNN architecture for network traffic classification
2. Helper classes for model components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Convolutional block for the Deep Concatenated 2D-CNN.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x_pooled = self.pool(x2)
        return x_pooled, torch.cat([x, x2], dim=1)  # Return pooled output and concatenated features

class DeepConcatenatedCNN(nn.Module):
    """
    Deep Concatenated 2D-CNN for network traffic classification.
    This architecture is designed to effectively learn patterns in network traffic data.
    """
    def __init__(self, input_size, num_classes):
        super(DeepConcatenatedCNN, self).__init__()
        
        # For CNN, we need to reshape the input to image-like format
        self.height = 16
        self.width = 16
        
        # Initial channel is 1 (grayscale)
        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(17, 32)  # 17 = 1 (original) + 16 (from block1)
        self.block3 = ConvBlock(49, 64)  # 49 = 17 + 32
        self.block4 = ConvBlock(113, 128)  # 113 = 49 + 64
        
        # Use an adaptive pooling layer for flexibility with input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Reshape input to image format if it's not already
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            feature_size = x.shape[1]
            
            # Pad or truncate features to fit the desired dimensions
            target_size = self.height * self.width
            if feature_size < target_size:
                # Pad with zeros
                padded = torch.zeros(batch_size, target_size, device=x.device)
                padded[:, :feature_size] = x
                x = padded
            elif feature_size > target_size:
                # Truncate
                x = x[:, :target_size]
            
            # Reshape to image format
            x = x.view(batch_size, 1, self.height, self.width)
        
        # Forward through convolutional blocks
        x1, concat1 = self.block1(x)
        x2, concat2 = self.block2(concat1)
        x3, concat3 = self.block3(concat2)
        x4, concat4 = self.block4(concat3)
        
        # Use adaptive pooling to get a fixed size output regardless of input dimensions
        x = self.adaptive_pool(x4)
        
        # Flatten and forward through fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class OpenSetClassifier:
    """
    Open-set classifier for detecting unknown attack patterns.
    This is a simplified version that uses a threshold-based approach.
    """
    def __init__(self, threshold=0.8):
        """
        Initialize the open-set classifier.
        
        Args:
            threshold: Confidence threshold for known classes
        """
        self.threshold = threshold
    
    def predict(self, probabilities):
        """
        Predict whether samples are known or unknown.
        
        Args:
            probabilities: Class probabilities from the closed-set classifier
            
        Returns:
            Boolean array (True for known, False for unknown)
        """
        # Get the maximum probability for each sample
        max_probs = probabilities.max(axis=1)
        
        # Samples with max probability below threshold are considered unknown
        return max_probs >= self.threshold

def save_model(model, path, metadata=None):
    """
    Save a PyTorch model with optional metadata.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        metadata: Dictionary of metadata to save with the model
    """
    if metadata is None:
        metadata = {}
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(path, model_class=DeepConcatenatedCNN, device=None):
    """
    Load a PyTorch model with metadata.
    
    Args:
        path: Path to the saved model
        model_class: Class of the model to load
        device: Device to load the model to
        
    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved dictionary
    save_dict = torch.load(path, map_location=device)
    
    # Extract metadata
    metadata = save_dict.get('metadata', {})
    
    # Create a new model instance
    input_size = metadata.get('input_size', 0)
    num_classes = metadata.get('num_classes', 2)
    model = model_class(input_size, num_classes).to(device)
    
    # Load the state dictionary
    model.load_state_dict(save_dict['model_state_dict'])
    
    return model, metadata
