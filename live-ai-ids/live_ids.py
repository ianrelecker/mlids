#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
live_ids.py - Main script for the Live AI-IDS system

This script provides a command-line interface to:
1. Run the live network intrusion detection system
2. Load a trained model and detect attacks in real-time
3. Display alerts and statistics
"""

import os
import sys
import time
import argparse
import signal
import torch
from src.utils import load_config, setup_logging, get_available_interfaces, get_system_info
from src.model import load_model, DeepConcatenatedCNN
from src.detector import Detector

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Live AI-IDS: Real-time Network Intrusion Detection System'
    )
    
    parser.add_argument(
        '--interface', '-i', type=str, default=None,
        help='Network interface to capture packets from (default: system default)'
    )
    
    parser.add_argument(
        '--model', '-m', type=str, default='models/final_model.pt',
        help='Path to the trained model (default: models/final_model.pt)'
    )
    
    parser.add_argument(
        '--config', '-c', type=str, default='config.json',
        help='Path to the configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--duration', '-d', type=int, default=0,
        help='Duration in seconds to run the detection (default: 0 for indefinite)'
    )
    
    parser.add_argument(
        '--threshold', '-t', type=float, default=None,
        help='Confidence threshold for attack detection (default: from config)'
    )
    
    parser.add_argument(
        '--list-interfaces', '-l', action='store_true',
        help='List available network interfaces and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit."""
    print("\nStopping detection...")
    if 'detector' in globals() and detector is not None:
        detector.stop_detection()
    print("Exiting...")
    sys.exit(0)

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
    if args.threshold:
        config['confidence_threshold'] = args.threshold
    
    # Set up logging
    logger = setup_logging(log_dir=os.path.join(config.get('results_dir', 'results'), 'logs'))
    
    # Print system information
    system_info = get_system_info()
    logger.info("System Information:")
    for key, value in system_info.items():
        if isinstance(value, str) and len(value) > 100:
            logger.info(f"  {key}: {value[:100]}...")
        else:
            logger.info(f"  {key}: {value}")
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.error("Please train a model first using train_model.py")
        return
    
    # Load the model
    try:
        logger.info(f"Loading model from {args.model}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, metadata = load_model(args.model, device=device)
        logger.info(f"Model loaded successfully. Device: {device}")
        
        # Log model metadata
        logger.info("Model metadata:")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create detector
    global detector
    detector = Detector(
        model=model,
        config=config,
        scaler=metadata.get('scaler'),
        label_encoder=metadata.get('label_encoder')
    )
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start detection
    logger.info("Starting network attack detection...")
    print("\n" + "="*60)
    print(" Live AI-IDS: Real-time Network Intrusion Detection System ")
    print("="*60)
    print(f"Interface: {config.get('interface') or 'default'}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {config.get('confidence_threshold', 0.8)}")
    if args.duration > 0:
        print(f"Duration: {args.duration} seconds")
    else:
        print("Duration: indefinite (press Ctrl+C to stop)")
    print("="*60 + "\n")
    
    detector.start_detection()
    
    # Run for specified duration or indefinitely
    try:
        if args.duration > 0:
            time.sleep(args.duration)
            detector.stop_detection()
        else:
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
                # Print statistics periodically
                if args.verbose and detector.total_packets > 0 and detector.total_packets % 1000 == 0:
                    stats = detector.get_statistics()
                    print("\nDetection Statistics:")
                    print(f"  Duration: {stats['duration']:.2f} seconds")
                    print(f"  Total packets: {stats['total_packets']}")
                    print(f"  Benign packets: {stats['benign_packets']} ({stats['benign_packets']/stats['total_packets']*100:.2f}%)")
                    print(f"  Attack packets: {stats['attack_packets']} ({stats['attack_packets']/stats['total_packets']*100:.2f}%)")
                    print(f"  Unknown packets: {stats['unknown_packets']} ({stats['unknown_packets']/stats['total_packets']*100:.2f}%)")
                    print(f"  Packets per second: {stats['packets_per_second']:.2f}")
    except KeyboardInterrupt:
        # This should be caught by the signal handler
        pass
    finally:
        # Make sure detection is stopped
        if detector:
            detector.stop_detection()
    
    # Print final statistics
    stats = detector.get_statistics()
    print("\nFinal Detection Statistics:")
    print(f"  Duration: {stats['duration']:.2f} seconds")
    print(f"  Total packets: {stats['total_packets']}")
    print(f"  Benign packets: {stats['benign_packets']} ({stats['benign_packets']/max(1, stats['total_packets'])*100:.2f}%)")
    print(f"  Attack packets: {stats['attack_packets']} ({stats['attack_packets']/max(1, stats['total_packets'])*100:.2f}%)")
    print(f"  Unknown packets: {stats['unknown_packets']} ({stats['unknown_packets']/max(1, stats['total_packets'])*100:.2f}%)")
    print(f"  Packets per second: {stats['packets_per_second']:.2f}")
    
    logger.info("Detection completed.")

if __name__ == "__main__":
    main()
