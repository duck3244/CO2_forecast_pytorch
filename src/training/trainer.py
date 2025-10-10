import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import time
from .metrics import calculate_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        if config['training']['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config['training']['scheduler_patience'],
                factor=config['training']['scheduler_factor'],
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate additional metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        metrics = calculate_metrics(targets, predictions)
        
        return avg_loss, metrics
    
    def train(self):
        """Full training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        epochs = self.config['training']['epochs']
        patience = self.config['training']['patience']
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val Metrics: {val_metrics}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"New best validation loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{patience}")
                
                if self.patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation loss: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Model loaded from {filepath}")
        return checkpoint

class EnsembleTrainer:
    """Trainer for ensemble models"""
    
    def __init__(self, individual_models, train_loader, val_loader, config, device='cpu'):
        self.individual_models = individual_models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Train individual models first
        self.trained_models = []
        self.train_individual_models()
        
        # Create ensemble
        from ..models.models import EnsembleModel
        self.ensemble_model = EnsembleModel(self.trained_models).to(device)
        
    def train_individual_models(self):
        """Train each individual model"""
        print("Training individual models for ensemble...")
        
        for i, model in enumerate(self.individual_models):
            print(f"\nTraining model {i + 1}/{len(self.individual_models)}")
            
            trainer = Trainer(model, self.train_loader, self.val_loader, self.config, self.device)
            trainer.train()
            
            self.trained_models.append(model)
            
    def fine_tune_ensemble(self, epochs=50):
        """Fine-tune ensemble weights"""
        print("Fine-tuning ensemble weights...")
        
        # Freeze individual model parameters
        for model in self.trained_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Only train ensemble weights
        optimizer = optim.Adam(
            [self.ensemble_model.weights],
            lr=0.01
        )
        
        criterion = nn.MSELoss()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.ensemble_model.train()
            total_loss = 0
            num_batches = 0
            
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.ensemble_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if epoch % 10 == 0:
                val_loss, _ = self.validate_ensemble()
                print(f"Epoch {epoch}: Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                print(f"Ensemble weights: {torch.softmax(self.ensemble_model.weights, dim=0).detach().cpu().numpy()}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
    
    def validate_ensemble(self):
        """Validate ensemble model"""
        self.ensemble_model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.ensemble_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate additional metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        metrics = calculate_metrics(targets, predictions)
        
        return avg_loss, metrics
        