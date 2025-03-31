#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trainer.py - Model training functionality

This module handles:
1. Training the neural network model
2. Evaluating model performance
3. Visualizing training progress and results
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from .model import save_model

class Trainer:
    """
    Handles training and evaluation of the model.
    """
    def __init__(self, model, config):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # Set device
        self.device = next(model.parameters()).device
        
    def train(self, train_loader, val_loader=None):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            
        Returns:
            losses: List of training losses
            accuracies: List of training accuracies
        """
        print("Starting training...")
        self.model.train()
        
        losses = []
        accuracies = []
        val_accuracies = []
        
        epochs = self.config.get('epochs', 150)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch statistics
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            # Validate if validation loader is provided
            val_accuracy = None
            if val_loader:
                val_accuracy = self.evaluate(val_loader)
                val_accuracies.append(val_accuracy)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%, Val Accuracy: {val_accuracy*100:.2f}%')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.config.get('model_dir', 'models'),
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                metadata = {
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'val_accuracy': val_accuracy,
                    'input_size': self.config.get('input_size'),
                    'num_classes': self.config.get('num_classes')
                }
                save_model(self.model, checkpoint_path, metadata)
        
        # Save final model
        model_path = os.path.join(
            self.config.get('model_dir', 'models'),
            'final_model.pt'
        )
        metadata = {
            'epochs': epochs,
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1],
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else None,
            'input_size': self.config.get('input_size'),
            'num_classes': self.config.get('num_classes'),
            'class_names': self.config.get('class_names', [])
        }
        save_model(self.model, model_path, metadata)
        
        # Plot training progress
        self.plot_training_progress(losses, accuracies, val_accuracies)
        
        return losses, accuracies
    
    def evaluate(self, test_loader):
        """
        Evaluate the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            accuracy: Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def get_predictions(self, test_loader):
        """
        Get predictions for all samples in the test loader.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Prediction probabilities
        """
        self.model.eval()
        y_true = []
        y_pred = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred), np.array(all_probs)
    
    def plot_training_progress(self, losses, accuracies, val_accuracies=None):
        """
        Plot training loss and accuracy over epochs.
        
        Args:
            losses: List of training losses
            accuracies: List of training accuracies
            val_accuracies: List of validation accuracies (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(losses, '-o', label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, '-o', label='Training Accuracy')
        if val_accuracies:
            plt.plot(val_accuracies, '-o', label='Validation Accuracy')
            plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'training_progress.png'))
        print(f"Training progress plot saved to {os.path.join(results_dir, 'training_progress.png')}")
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save the plot
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix plot saved to {os.path.join(results_dir, 'confusion_matrix.png')}")
        
        plt.close()
    
    def print_classification_report(self, y_true, y_pred, class_names):
        """
        Print classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            report: Classification report as string
        """
        # Get unique classes in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # If there's only one class, we need to handle it specially
        if len(unique_classes) == 1:
            print("Classification Report:")
            print(f"Only one class present in the test set: {class_names[unique_classes[0]]}")
            report = f"Only one class present: {class_names[unique_classes[0]]}\n"
            report += f"Accuracy: {accuracy_score(y_true, y_pred):.2f}"
        else:
            # Use the labels parameter to specify which labels to include in the report
            report = classification_report(
                y_true, 
                y_pred, 
                labels=unique_classes,
                target_names=[class_names[i] for i in unique_classes], 
                zero_division=0
            )
            print("Classification Report:")
            print(report)
        
        # Save the report
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        return report
