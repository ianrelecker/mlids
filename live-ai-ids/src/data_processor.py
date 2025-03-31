#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_processor.py - Network packet capture and feature extraction

This module handles:
1. Capturing live network packets using Scapy
2. Extracting relevant features from packets
3. Preprocessing features for model input
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scapy.all as scapy
import ipaddress
import time
from datetime import datetime
import threading
import queue
import torch
from torch.utils.data import DataLoader, TensorDataset

class PacketCapture:
    """
    Captures live network packets and extracts features.
    """
    def __init__(self, interface=None, max_packets=10000, buffer_size=1000):
        """
        Initialize the packet capture module.
        
        Args:
            interface: Network interface to capture from (None for default)
            max_packets: Maximum number of packets to capture
            buffer_size: Size of the packet buffer
        """
        self.interface = interface
        self.max_packets = max_packets
        self.buffer_size = buffer_size
        self.packet_buffer = queue.Queue(maxsize=buffer_size)
        self._stop_flag = threading.Event()
        self.capture_thread = None
        self.packet_count = 0
        self.start_time = None
        
    def start_capture(self):
        """Start capturing packets in a separate thread."""
        if self.capture_thread and self.capture_thread.is_alive():
            print("Packet capture already running.")
            return
        
        self._stop_flag.clear()
        self.packet_count = 0
        self.start_time = time.time()
        
        self.capture_thread = threading.Thread(target=self._capture_packets)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print(f"Started packet capture on interface: {self.interface or 'default'}")
        
    def stop_capture(self):
        """Stop the packet capture thread."""
        if not self.capture_thread or not self.capture_thread.is_alive():
            print("No packet capture running.")
            return
        
        self._stop_flag.set()
        self.capture_thread.join(timeout=2.0)
        
        duration = time.time() - self.start_time
        print(f"Stopped packet capture. Captured {self.packet_count} packets in {duration:.2f} seconds.")
        
    def _capture_packets(self):
        """Internal method to capture packets using Scapy."""
        def packet_callback(packet):
            if self._stop_flag.is_set() or self.packet_count >= self.max_packets:
                return
            
            try:
                if not self.packet_buffer.full():
                    self.packet_buffer.put(packet, block=False)
                    self.packet_count += 1
                    
                    if self.packet_count % 100 == 0:
                        print(f"Captured {self.packet_count} packets...")
                        
                    if self.packet_count >= self.max_packets:
                        print(f"Reached maximum packet count ({self.max_packets})")
                        self._stop_flag.set()
            except queue.Full:
                pass  # Buffer is full, skip this packet
        
        try:
            scapy.sniff(
                iface=self.interface,
                prn=packet_callback,
                store=False,
                stop_filter=lambda _: self._stop_flag.is_set()
            )
        except Exception as e:
            print(f"Error in packet capture: {e}")
            self._stop_flag.set()
    
    def get_packets(self, count=None, timeout=1.0):
        """
        Get packets from the buffer.
        
        Args:
            count: Number of packets to get (None for all available)
            timeout: Timeout in seconds for getting packets
            
        Returns:
            List of packets
        """
        packets = []
        start_time = time.time()
        
        try:
            while (count is None or len(packets) < count) and time.time() - start_time < timeout:
                try:
                    packet = self.packet_buffer.get(block=True, timeout=0.1)
                    packets.append(packet)
                    self.packet_buffer.task_done()
                except queue.Empty:
                    if self._stop_flag.is_set() and self.packet_buffer.empty():
                        break
        except Exception as e:
            print(f"Error getting packets: {e}")
        
        return packets

class FeatureExtractor:
    """
    Extracts features from network packets for intrusion detection.
    """
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_features(self, packets, label='BENIGN'):
        """
        Extract features from a list of packets.
        
        Args:
            packets: List of Scapy packets
            label: Label to assign to the packets
            
        Returns:
            DataFrame with extracted features
        """
        if not packets:
            return pd.DataFrame()
        
        features_list = []
        
        for packet in packets:
            features = self._extract_packet_features(packet)
            if features:
                features_list.append(features)
        
        if not features_list:
            return pd.DataFrame()
        
        # Create DataFrame with features
        columns = list(features_list[0].keys())
        data = pd.DataFrame(features_list, columns=columns)
        
        # Add label column
        data['Label'] = label
        
        return data
    
    def _extract_packet_features(self, packet):
        """
        Extract features from a single packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic packet information
        features['pkt_size'] = len(packet)
        features['has_payload'] = 1 if scapy.Raw in packet else 0
        
        # Ethernet layer features
        if scapy.Ether in packet:
            features['eth_type'] = int(packet[scapy.Ether].type)
        else:
            features['eth_type'] = 0
        
        # IP layer features
        if scapy.IP in packet:
            ip = packet[scapy.IP]
            features['ip_version'] = ip.version
            features['ip_ihl'] = ip.ihl
            features['ip_tos'] = ip.tos
            features['ip_len'] = ip.len
            features['ip_id'] = ip.id
            features['ip_flags'] = ip.flags
            features['ip_frag'] = ip.frag
            features['ip_ttl'] = ip.ttl
            features['ip_proto'] = ip.proto
            features['ip_chksum'] = ip.chksum
            
            # Convert IP addresses to numerical values
            try:
                src_ip = int(ipaddress.IPv4Address(ip.src))
                dst_ip = int(ipaddress.IPv4Address(ip.dst))
                features['ip_src'] = src_ip
                features['ip_dst'] = dst_ip
            except:
                features['ip_src'] = 0
                features['ip_dst'] = 0
        else:
            # Default values if IP layer is not present
            features['ip_version'] = 0
            features['ip_ihl'] = 0
            features['ip_tos'] = 0
            features['ip_len'] = 0
            features['ip_id'] = 0
            features['ip_flags'] = 0
            features['ip_frag'] = 0
            features['ip_ttl'] = 0
            features['ip_proto'] = 0
            features['ip_chksum'] = 0
            features['ip_src'] = 0
            features['ip_dst'] = 0
        
        # TCP layer features
        if scapy.TCP in packet:
            tcp = packet[scapy.TCP]
            features['tcp_sport'] = tcp.sport
            features['tcp_dport'] = tcp.dport
            features['tcp_seq'] = tcp.seq
            features['tcp_ack'] = tcp.ack
            features['tcp_dataofs'] = tcp.dataofs
            features['tcp_reserved'] = tcp.reserved
            features['tcp_flags'] = int(tcp.flags)
            features['tcp_window'] = tcp.window
            features['tcp_chksum'] = tcp.chksum
            features['tcp_urgptr'] = tcp.urgptr
            features['is_tcp'] = 1
            features['is_udp'] = 0
        else:
            # Default values if TCP layer is not present
            features['tcp_sport'] = 0
            features['tcp_dport'] = 0
            features['tcp_seq'] = 0
            features['tcp_ack'] = 0
            features['tcp_dataofs'] = 0
            features['tcp_reserved'] = 0
            features['tcp_flags'] = 0
            features['tcp_window'] = 0
            features['tcp_chksum'] = 0
            features['tcp_urgptr'] = 0
            features['is_tcp'] = 0
        
        # UDP layer features
        if scapy.UDP in packet:
            udp = packet[scapy.UDP]
            features['udp_sport'] = udp.sport
            features['udp_dport'] = udp.dport
            features['udp_len'] = udp.len
            features['udp_chksum'] = udp.chksum
            features['is_udp'] = 1
        else:
            # Default values if UDP layer is not present
            features['udp_sport'] = 0
            features['udp_dport'] = 0
            features['udp_len'] = 0
            features['udp_chksum'] = 0
            features['is_udp'] = 0
        
        # ICMP layer features
        if scapy.ICMP in packet:
            icmp = packet[scapy.ICMP]
            features['icmp_type'] = icmp.type
            features['icmp_code'] = icmp.code
            features['icmp_chksum'] = icmp.chksum
            features['is_icmp'] = 1
        else:
            # Default values if ICMP layer is not present
            features['icmp_type'] = 0
            features['icmp_code'] = 0
            features['icmp_chksum'] = 0
            features['is_icmp'] = 0
        
        # Payload features (if present)
        if scapy.Raw in packet:
            payload = packet[scapy.Raw].load
            features['payload_len'] = len(payload)
            
            # Calculate entropy of payload (measure of randomness)
            try:
                entropy = 0
                for i in range(256):
                    p_i = payload.count(i) / len(payload)
                    if p_i > 0:
                        entropy -= p_i * np.log2(p_i)
                features['payload_entropy'] = entropy
            except:
                features['payload_entropy'] = 0
            
            # Count printable characters in payload
            printable_count = sum(c >= 32 and c <= 126 for c in payload)
            features['payload_printable_ratio'] = printable_count / len(payload) if len(payload) > 0 else 0
        else:
            features['payload_len'] = 0
            features['payload_entropy'] = 0
            features['payload_printable_ratio'] = 0
        
        # Time-based features
        features['timestamp'] = packet.time if hasattr(packet, 'time') else 0
        
        return features

class DataProcessor:
    """
    Handles loading, preprocessing, and partitioning of network data.
    """
    def __init__(self, config):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.packet_capture = PacketCapture(
            interface=config.get('interface'),
            max_packets=config.get('max_packets', 10000),
            buffer_size=config.get('buffer_size', 1000)
        )
    
    def capture_training_data(self, duration=60, label='BENIGN', save_path=None):
        """
        Capture network data for training.
        
        Args:
            duration: Duration in seconds to capture
            label: Label to assign to the captured data
            save_path: Path to save the captured data (optional)
            
        Returns:
            DataFrame with captured data
        """
        print(f"Capturing training data for {duration} seconds...")
        
        self.packet_capture.start_capture()
        
        # Wait for the specified duration
        time.sleep(duration)
        
        # Stop capture
        self.packet_capture.stop_capture()
        
        # Get all captured packets
        packets = self.packet_capture.get_packets()
        
        # Extract features
        data = self.feature_extractor.extract_features(packets, label=label)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.to_csv(save_path, index=False)
            print(f"Saved training data to {save_path}")
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the dataset by cleaning and transforming.
        
        Args:
            data: DataFrame with network data
            
        Returns:
            Preprocessed DataFrame
        """
        print('Preprocessing data...')
        
        # Handle infinity values and non-numeric values
        # First, ensure all columns are numeric
        for col in data.columns:
            if col != 'Label':  # Skip the Label column
                try:
                    # Try to convert to float, which can handle NaN values
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to numeric: {e}")
        
        # Now replace infinity values
        data.replace([float('inf'), float('-inf')], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        # Cap classes to specific sample size for balance
        samples_per_class = self.config.get('samples_per_class', 5000)
        print(f'Capping each class to {samples_per_class} samples...')
        data_capped = data.groupby('Label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42)
        ).reset_index(drop=True)
        
        print(f"Data after preprocessing: {data_capped.shape}")
        return data_capped
    
    def partition_data(self, data):
        """
        Partition the data into training, validation, and test sets.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            Dictionary with partitioned data
        """
        print('Partitioning data...')
        
        # Separate features and labels
        X = data.drop(columns=['Label']).values
        y = data['Label'].values
        
        # Print class distribution
        print("Class distribution:")
        for label, count in data['Label'].value_counts().items():
            print(f"  {label}: {count} samples")
        
        # If there's only one class, create a synthetic second class
        if len(data['Label'].unique()) == 1:
            print("\nWARNING: Only one class detected in data.")
            print("Creating synthetic 'ATTACK' class for training purposes.")
            
            # Create a copy of 10% of the data and label it as 'ATTACK'
            attack_indices = np.random.choice(len(X), size=int(len(X) * 0.1), replace=False)
            X_attack = X[attack_indices].copy()
            
            # Add some noise to make the attack class different
            X_attack += np.random.normal(0, 0.1, X_attack.shape)
            
            # Combine original and synthetic data
            X = np.vstack([X, X_attack])
            y = np.concatenate([y, np.array(['ATTACK'] * len(X_attack))])
            
            print(f"Added {len(X_attack)} synthetic attack samples.")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print("\nEncoded class mapping:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {i}: {class_name}")
        
        # Split into training and test sets
        test_size = self.config.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Further split training data into training and validation
        val_size = self.config.get('val_size', 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Get device for PyTorch tensors
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU) device for training")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device for training")
        else:
            device = torch.device("cpu")
            print("Using CPU device for training")
        
        # Create PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 256)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_size': X_train.shape[1],
            'num_classes': len(label_encoder.classes_),
            'class_names': label_encoder.classes_,
            'scaler': self.scaler,
            'label_encoder': label_encoder
        }
    
    def process_live_packets(self, packets, model, scaler=None, label_encoder=None):
        """
        Process live packets for detection.
        
        Args:
            packets: List of Scapy packets
            model: Trained PyTorch model
            scaler: Fitted StandardScaler (optional)
            label_encoder: Fitted LabelEncoder (optional)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not packets:
            return [], []
        
        # Extract features
        data = self.feature_extractor.extract_features(packets)
        if data.empty:
            return [], []
        
        # Prepare features
        X = data.drop(columns=['Label'])
        
        # Ensure all columns are numeric
        for col in X.columns:
            try:
                # Try to convert to float, which can handle NaN values
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {e}")
        
        # Replace infinity values
        X.replace([float('inf'), float('-inf')], np.nan, inplace=True)
        
        # Fill NaN values with 0
        X.fillna(0, inplace=True)
        
        # Convert to numpy array
        X_values = X.values
        
        # Scale features if scaler is provided
        if scaler:
            try:
                X_values = scaler.transform(X_values)
            except Exception as e:
                print(f"Warning: Error in scaling features: {e}")
                # If scaling fails, just use the unscaled features
                pass
        
        # Convert to PyTorch tensor
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X_tensor = torch.tensor(X_values, dtype=torch.float32).to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Convert predictions to class names if label_encoder is provided
        predictions = predicted.cpu().numpy()
        if label_encoder:
            predictions = label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities.cpu().numpy()
