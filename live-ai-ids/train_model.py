#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py - Script to train a new model for the Live AI-IDS system

This script provides a command-line interface to:
1. Capture network traffic for training
2. Train a model on the captured data
3. Save the trained model for use with the live detection system
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import pandas as pd
from src.utils import load_config, setup_logging, get_available_interfaces
from src.data_processor import DataProcessor
from src.model import DeepConcatenatedCNN, save_model
from src.trainer import Trainer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a new model for the Live AI-IDS system'
    )
    
    parser.add_argument(
        '--interface', '-i', type=str, default=None,
        help='Network interface to capture packets from (default: system default)'
    )
    
    parser.add_argument(
        '--config', '-c', type=str, default='config.json',
        help='Path to the configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--duration', '-d', type=int, default=300,
        help='Duration in seconds to capture training data (default: 300)'
    )
    
    parser.add_argument(
        '--benign-label', '-b', type=str, default='BENIGN',
        help='Label for benign traffic (default: BENIGN)'
    )
    
    parser.add_argument(
        '--attack-label', '-a', type=str, default='ATTACK',
        help='Label for attack traffic (default: ATTACK)'
    )
    
    parser.add_argument(
        '--attack-duration', type=int, default=0,
        help='Duration in seconds to capture attack traffic (default: 0, no attack capture)'
    )
    
    parser.add_argument(
        '--save-data', '-s', action='store_true',
        help='Save captured data to CSV files'
    )
    
    parser.add_argument(
        '--load-data', '-l', type=str, default=None,
        help='Load data from CSV file instead of capturing'
    )
    
    parser.add_argument(
        '--epochs', '-e', type=int, default=None,
        help='Number of training epochs (default: from config)'
    )
    
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size for training (default: from config)'
    )
    
    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Learning rate for training (default: from config)'
    )
    
    parser.add_argument(
        '--model-path', '-m', type=str, default=None,
        help='Path to save the trained model (default: from config)'
    )
    
    parser.add_argument(
        '--list-interfaces', action='store_true',
        help='List available network interfaces and exit'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # List interfaces if requested
    if args.list_interfaces:
        print("Available network interfaces:")
        for interface in get_available_interfaces():
            print(f"  - {interface}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.interface:
        config['interface'] = args.interface
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.model_path:
        config['model_save_path'] = args.model_path
    
    # Set up logging
    logger = setup_logging(log_dir=os.path.join(config.get('results_dir', 'results'), 'logs'))
    logger.info("Starting model training...")
    
    # Create data processor
    data_processor = DataProcessor(config)
    
    # Get training data
    if args.load_data:
        # Load data from CSV file
        logger.info(f"Loading data from {args.load_data}...")
        try:
            data = pd.read_csv(args.load_data)
            logger.info(f"Loaded data with shape: {data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return
    else:
        # Capture benign traffic
        logger.info(f"Capturing benign traffic for {args.duration} seconds...")
        print(f"\nCapturing benign traffic for {args.duration} seconds...")
        print("Please generate normal network traffic during this time.")
        print("For example, browse websites, check email, etc.\n")
        
        benign_data = data_processor.capture_training_data(
            duration=args.duration,
            label=args.benign_label,
            save_path=os.path.join(config.get('data_dir', 'data'), 'benign_traffic.csv') if args.save_data else None
        )
        
        # Capture attack traffic if requested
        attack_data = None
        if args.attack_duration > 0:
            logger.info(f"Capturing attack traffic for {args.attack_duration} seconds...")
            print(f"\nCapturing attack traffic for {args.attack_duration} seconds...")
            print("Please generate attack-like traffic during this time.")
            print("For example, run port scans, excessive connection attempts, etc.\n")
            
            attack_data = data_processor.capture_training_data(
                duration=args.attack_duration,
                label=args.attack_label,
                save_path=os.path.join(config.get('data_dir', 'data'), 'attack_traffic.csv') if args.save_data else None
            )
        
        # Combine data
        if attack_data is not None and not attack_data.empty:
            data = pd.concat([benign_data, attack_data], ignore_index=True)
        else:
            data = benign_data
        
        # Save combined data if requested
        if args.save_data:
            combined_path = os.path.join(config.get('data_dir', 'data'), 'combined_traffic.csv')
            data.to_csv(combined_path, index=False)
            logger.info(f"Combined data saved to {combined_path}")
    
    # Check if we have enough data
    if data.empty:
        logger.error("No data captured or loaded. Exiting.")
        return
    
    # Preprocess data
    logger.info("Preprocessing data...")
    data = data_processor.preprocess_data(data)
    
    # Partition data
    logger.info("Partitioning data...")
    data_partitions = data_processor.partition_data(data)
    
    # Update config with data information
    config['input_size'] = data_partitions['input_size']
    config['num_classes'] = data_partitions['num_classes']
    config['class_names'] = data_partitions['class_names'].tolist()
    
    # Create model
    logger.info("Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepConcatenatedCNN(
        input_size=data_partitions['input_size'],
        num_classes=data_partitions['num_classes']
    ).to(device)
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Train model
    logger.info("Training model...")
    print("\nTraining model...")
    losses, accuracies = trainer.train(
        data_partitions['train_loader'],
        data_partitions['val_loader']
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    print("\nEvaluating model...")
    accuracy = trainer.evaluate(data_partitions['test_loader'])
    logger.info(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Test accuracy: {accuracy*100:.2f}%")
    
    # Get predictions for visualization
    y_true, y_pred, probabilities = trainer.get_predictions(data_partitions['test_loader'])
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(y_true, y_pred, data_partitions['class_names'])
    
    # Print classification report
    trainer.print_classification_report(y_true, y_pred, data_partitions['class_names'])
    
    # Save model with metadata
    model_path = config.get('model_save_path', os.path.join(config.get('model_dir', 'models'), 'final_model.pt'))
    metadata = {
        'input_size': data_partitions['input_size'],
        'num_classes': data_partitions['num_classes'],
        'class_names': data_partitions['class_names'].tolist(),
        'accuracy': accuracy,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config
    }
    save_model(model, model_path, metadata)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler and label_encoder separately
    import pickle
    scaler_path = os.path.join(config.get('model_dir', 'models'), 'scaler.pkl')
    label_encoder_path = os.path.join(config.get('model_dir', 'models'), 'label_encoder.pkl')
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(data_processor.scaler, f)
    
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(data_partitions['label_encoder'], f)
    
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Label encoder saved to {label_encoder_path}")
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to {model_path}")
    print(f"You can now use the model for live detection with: python live_ids.py")

if __name__ == "__main__":
    main()
