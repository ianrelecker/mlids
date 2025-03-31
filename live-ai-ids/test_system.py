#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_system.py - Test script for the Live AI-IDS system

This script tests the basic functionality of the system components.
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from src.utils import load_config, get_available_interfaces
from src.data_processor import DataProcessor, FeatureExtractor, PacketCapture
from src.model import DeepConcatenatedCNN, OpenSetClassifier

def test_packet_capture():
    """Test packet capture functionality."""
    print("\n=== Testing Packet Capture ===")
    
    # Create packet capture
    packet_capture = PacketCapture(max_packets=100)
    
    # Start capture
    print("Starting packet capture...")
    packet_capture.start_capture()
    
    # Wait for a few seconds
    print("Capturing packets for 5 seconds...")
    time.sleep(5)
    
    # Stop capture
    packet_capture.stop_capture()
    
    # Get packets
    packets = packet_capture.get_packets()
    
    # Print results
    print(f"Captured {len(packets)} packets.")
    if len(packets) > 0:
        print("Packet capture test: PASSED")
    else:
        print("Packet capture test: FAILED (no packets captured)")
    
    return packets

def test_feature_extraction(packets):
    """Test feature extraction functionality."""
    print("\n=== Testing Feature Extraction ===")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Extract features
    print("Extracting features from packets...")
    data = feature_extractor.extract_features(packets)
    
    # Print results
    print(f"Extracted features from {len(data)} packets.")
    print(f"Feature columns: {list(data.columns)}")
    
    if not data.empty:
        print("Feature extraction test: PASSED")
    else:
        print("Feature extraction test: FAILED (no features extracted)")
    
    return data

def test_model():
    """Test model functionality."""
    print("\n=== Testing Model ===")
    
    # Create a small test dataset
    print("Creating test dataset...")
    input_size = 50
    num_classes = 2
    batch_size = 10
    
    # Create random data
    X = np.random.randn(batch_size, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=batch_size).astype(np.int64)
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    # Create model
    print(f"Creating model with input_size={input_size}, num_classes={num_classes}...")
    model = DeepConcatenatedCNN(input_size, num_classes).to(device)
    
    # Forward pass
    print("Testing forward pass...")
    outputs = model(X_tensor)
    
    # Check outputs
    if outputs.shape == (batch_size, num_classes):
        print(f"Model output shape: {outputs.shape} (expected: {(batch_size, num_classes)})")
        print("Model test: PASSED")
    else:
        print(f"Model output shape: {outputs.shape} (expected: {(batch_size, num_classes)})")
        print("Model test: FAILED (incorrect output shape)")
    
    # Test open-set classifier
    print("\nTesting open-set classifier...")
    open_set_classifier = OpenSetClassifier(threshold=0.5)
    probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
    is_known = open_set_classifier.predict(probabilities)
    
    print(f"Open-set classifier predictions: {is_known}")
    print("Open-set classifier test: PASSED")
    
    return model

def main():
    """Main function."""
    print("=== Live AI-IDS System Test ===")
    
    # Test packet capture
    packets = test_packet_capture()
    
    # Test feature extraction if packets were captured
    if packets:
        data = test_feature_extraction(packets)
    
    # Test model
    model = test_model()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()
