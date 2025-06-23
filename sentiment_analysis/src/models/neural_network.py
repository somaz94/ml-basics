"""
Neural network model for English sentiment analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score

from ..config import NN_CONFIG


class SentimentNN(nn.Module):
    """Neural network for sentiment analysis"""
    
    def __init__(self, input_size: int, config: dict = NN_CONFIG):
        super(SentimentNN, self).__init__()
        
        hidden_sizes = config['hidden_sizes']
        dropout_rate = config['dropout_rate']
        num_classes = config['num_classes']
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkTrainer:
    """Neural network trainer"""
    
    def __init__(self, input_size: int, config: dict = NN_CONFIG):
        self.config = config
        self.model = SentimentNN(input_size, config)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate']
        )
        self.training_losses = []
    
    def _prepare_tensors(self, X_train, X_test, y_train, y_test) -> Tuple[torch.Tensor, ...]:
        """Prepare tensors for training"""
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test.values if hasattr(y_test, 'values') else y_test)
        
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def train(self, X_train, X_test, y_train, y_test) -> dict:
        """Train neural network"""
        print("Training Neural Network...")
        
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = self._prepare_tensors(
            X_train, X_test, y_train, y_test
        )
        
        self.model.train()
        for epoch in range(self.config['epochs']):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            
            self.training_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], Loss: {loss.item():.4f}')
        
        return self.evaluate(X_test_tensor, y_test_tensor, y_test)
    
    def evaluate(self, X_test_tensor, y_test_tensor, y_test_original) -> dict:
        """Evaluate model"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(y_test_original, predicted.numpy())
            
            result = {
                'accuracy': accuracy,
                'predictions': predicted.numpy(),
                'model': self.model,
                'training_losses': self.training_losses
            }
            
            print(f"Neural Network Accuracy: {accuracy:.3f}")
            return result
    
    def predict_single(self, features) -> int:
        """Predict single sample"""
        self.model.eval()
        with torch.no_grad():
            if hasattr(features, 'toarray'):
                features = features.toarray()
            
            features_tensor = torch.FloatTensor(features)
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()