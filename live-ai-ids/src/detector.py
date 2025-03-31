#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detector.py - Attack detection logic

This module handles:
1. Real-time detection of network attacks
2. Classification of traffic as benign or malicious
3. Alerting when attacks are detected
"""

import time
import threading
import queue
import numpy as np
import torch
import scapy.all as scapy
from .model import OpenSetClassifier
from .data_processor import DataProcessor

class Detector:
    """
    Detects network attacks in real-time.
    """
    def __init__(self, model, config, scaler=None, label_encoder=None):
        """
        Initialize the detector.
        
        Args:
            model: Trained PyTorch model
            config: Configuration dictionary
            scaler: Fitted StandardScaler (optional)
            label_encoder: Fitted LabelEncoder (optional)
        """
        self.model = model
        self.config = config
        self.scaler = scaler
        self.label_encoder = label_encoder
        
        # Set up data processor
        self.data_processor = DataProcessor(config)
        
        # Set up open-set classifier
        self.open_set_classifier = OpenSetClassifier(
            threshold=config.get('confidence_threshold', 0.8)
        )
        
        # Set up detection thread
        self.detection_thread = None
        self._stop_flag = threading.Event()
        self.alert_queue = queue.Queue()
        
        # Detection statistics
        self.total_packets = 0
        self.benign_packets = 0
        self.attack_packets = 0
        self.unknown_packets = 0
        self.start_time = None
        
        # Set device
        self.device = next(model.parameters()).device
        print(f"Using device: {self.device}")
    
    def start_detection(self):
        """Start the detection process in a separate thread."""
        if self.detection_thread and self.detection_thread.is_alive():
            print("Detection already running.")
            return
        
        self._stop_flag.clear()
        self.total_packets = 0
        self.benign_packets = 0
        self.attack_packets = 0
        self.unknown_packets = 0
        self.start_time = time.time()
        
        # Start packet capture
        self.data_processor.packet_capture.start_capture()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        print("Started network attack detection.")
    
    def stop_detection(self):
        """Stop the detection process."""
        if not self.detection_thread or not self.detection_thread.is_alive():
            print("No detection running.")
            return
        
        self._stop_flag.set()
        self.data_processor.packet_capture.stop_capture()
        self.detection_thread.join(timeout=2.0)
        
        duration = time.time() - self.start_time
        print(f"Stopped detection after {duration:.2f} seconds.")
        print(f"Processed {self.total_packets} packets:")
        print(f"  - Benign: {self.benign_packets} ({self.benign_packets/max(1, self.total_packets)*100:.2f}%)")
        print(f"  - Attack: {self.attack_packets} ({self.attack_packets/max(1, self.total_packets)*100:.2f}%)")
        print(f"  - Unknown: {self.unknown_packets} ({self.unknown_packets/max(1, self.total_packets)*100:.2f}%)")
    
    def _detection_loop(self):
        """Internal method for the detection loop."""
        batch_size = self.config.get('detection_batch_size', 32)
        detection_interval = self.config.get('detection_interval', 1.0)  # seconds
        
        while not self._stop_flag.is_set():
            # Get packets from the buffer
            packets = self.data_processor.packet_capture.get_packets(
                count=batch_size,
                timeout=detection_interval
            )
            
            if not packets:
                continue
            
            # Process packets
            predictions, probabilities = self.data_processor.process_live_packets(
                packets,
                self.model,
                self.scaler,
                self.label_encoder
            )
            
            if len(predictions) == 0:
                continue
            
            # Detect unknown attacks using open-set classifier
            is_known = self.open_set_classifier.predict(probabilities)
            
            # Update statistics
            self.total_packets += len(predictions)
            
            # Process each prediction
            for i, (pred, known) in enumerate(zip(predictions, is_known)):
                if not known:
                    # Unknown attack
                    self.unknown_packets += 1
                    self._handle_alert("UNKNOWN", packets[i], probabilities[i])
                elif pred != "BENIGN":
                    # Known attack
                    self.attack_packets += 1
                    self._handle_alert(pred, packets[i], probabilities[i])
                else:
                    # Benign traffic
                    self.benign_packets += 1
            
            # Sleep for a short time to avoid CPU overuse
            time.sleep(0.01)
    
    def _handle_alert(self, attack_type, packet, probabilities):
        """
        Handle an attack alert.
        
        Args:
            attack_type: Type of attack detected
            packet: The packet that triggered the alert
            probabilities: Prediction probabilities
        """
        # Create alert message
        if hasattr(packet, 'time'):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(packet.time))
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Try to get source and destination addresses
        src_ip = "Unknown"
        dst_ip = "Unknown"
        
        # First try to get MAC addresses (always present in Ethernet frames)
        if hasattr(packet, 'src'):
            src_ip = packet.src
        if hasattr(packet, 'dst'):
            dst_ip = packet.dst
        
        # Try to get IP addresses if available
        if scapy.IP in packet:
            src_ip = packet[scapy.IP].src
            dst_ip = packet[scapy.IP].dst
        elif scapy.IPv6 in packet:
            src_ip = packet[scapy.IPv6].src
            dst_ip = packet[scapy.IPv6].dst
        
        alert = {
            'timestamp': timestamp,
            'attack_type': attack_type,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'confidence': float(np.max(probabilities)) if attack_type != "UNKNOWN" else 1.0 - float(np.max(probabilities)),
            'packet': packet
        }
        
        # Add to alert queue
        self.alert_queue.put(alert)
        
        # Print alert
        print(f"\n[ALERT] {timestamp} - {attack_type} attack detected!")
        print(f"  Source: {src_ip}")
        print(f"  Destination: {dst_ip}")
        print(f"  Confidence: {alert['confidence']:.2f}")
        print(f"  Packet: {packet.summary()}")
    
    def get_alerts(self, max_alerts=None):
        """
        Get alerts from the queue.
        
        Args:
            max_alerts: Maximum number of alerts to get (None for all)
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        try:
            while (max_alerts is None or len(alerts) < max_alerts) and not self.alert_queue.empty():
                alert = self.alert_queue.get(block=False)
                alerts.append(alert)
                self.alert_queue.task_done()
        except queue.Empty:
            pass
        
        return alerts
    
    def get_statistics(self):
        """
        Get detection statistics.
        
        Returns:
            Dictionary of statistics
        """
        duration = time.time() - (self.start_time or time.time())
        
        return {
            'duration': duration,
            'total_packets': self.total_packets,
            'benign_packets': self.benign_packets,
            'attack_packets': self.attack_packets,
            'unknown_packets': self.unknown_packets,
            'packets_per_second': self.total_packets / max(1, duration)
        }
