#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py - Utility functions for the Live AI-IDS system

This module provides:
1. Configuration management
2. Logging utilities
3. Helper functions for the system
"""

import os
import json
import logging
import time
import platform
import subprocess
from datetime import datetime

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """
    Set up logging for the system.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'live_ids_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('live_ai_ids')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'interface': None,  # None for default interface
        'max_packets': 10000,
        'buffer_size': 1000,
        'samples_per_class': 5000,
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 50,
        'test_size': 0.2,
        'val_size': 0.2,
        'confidence_threshold': 0.8,
        'detection_batch_size': 32,
        'detection_interval': 1.0,
        'model_dir': 'models',
        'data_dir': 'data',
        'results_dir': 'results'
    }
    
    # Try to load configuration from file
    config = default_config.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
    else:
        print(f"Configuration file {config_path} not found. Using default configuration.")
        
        # Save default configuration
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Default configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving default configuration: {e}")
    
    return config

def save_config(config, config_path='config.json'):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def get_available_interfaces():
    """
    Get a list of available network interfaces.
    
    Returns:
        List of interface names
    """
    import scapy.all as scapy
    return scapy.get_if_list()

def get_system_info():
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'hostname': platform.node()
    }
    
    # Try to get more detailed information
    try:
        if platform.system() == 'Linux':
            # Get CPU info
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
            info['cpu_info'] = cpu_info
            
            # Get memory info
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()
            info['mem_info'] = mem_info
        elif platform.system() == 'Darwin':  # macOS
            # Get CPU info
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            info['cpu_info'] = cpu_info
            
            # Get memory info
            mem_info = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode('utf-8').strip()
            info['mem_info'] = f"{int(mem_info) / (1024**3):.2f} GB"
        elif platform.system() == 'Windows':
            # Get CPU info
            cpu_info = subprocess.check_output(['wmic', 'cpu', 'get', 'name']).decode('utf-8').strip()
            info['cpu_info'] = cpu_info
            
            # Get memory info
            mem_info = subprocess.check_output(['wmic', 'computersystem', 'get', 'totalphysicalmemory']).decode('utf-8').strip()
            info['mem_info'] = f"{int(mem_info.split()[1]) / (1024**3):.2f} GB"
    except Exception as e:
        info['error'] = f"Error getting detailed system information: {e}"
    
    return info

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def format_bytes(bytes_value):
    """
    Format bytes to a human-readable string.
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Formatted string
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.2f} KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/(1024**2):.2f} MB"
    else:
        return f"{bytes_value/(1024**3):.2f} GB"
