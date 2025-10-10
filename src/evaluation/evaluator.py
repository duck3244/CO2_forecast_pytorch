import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..training.metrics import calculate_metrics, print_metrics
import os

class ModelEvaluator:
    def __init__(self, model, data_loader, device='cpu', scaler=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.scaler = scaler
        
    def predict(self):
        """Generate predictions"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        return predictions, targets
    
    def evaluate(self):
        """Evaluate model performance"""
        predictions, targets = self.predict()
        
        # If scaler is provided, inverse transform
        if self.scaler:
            predictions_orig = self.scaler.inverse_transform_co2(predictions)
            targets_orig = self.scaler.inverse_transform_co2(targets)
        else:
            predictions_orig = predictions
            targets_orig = targets
        
        # Calculate metrics
        metrics = calculate_metrics(targets_orig, predictions_orig)
        
        return {
            'metrics': metrics,
            'predictions': predictions_orig,
            'targets': targets_orig,
            'predictions_scaled': predictions,
            'targets_scaled': targets
        }
    
    def plot_predictions(self, results, save_path=None, title="Model Predictions"):
        """Plot predictions vs actual values"""
        predictions = results['predictions']
        targets = results['targets']
        
        # Flatten for plotting
        if len(predictions.shape) > 1:
            # For multi-step forecasting, take the first step
            pred_plot = predictions[:, 0] if predictions.shape[1] > 1 else predictions.flatten()
            target_plot = targets[:, 0] if targets.shape[1] > 1 else targets.flatten()
        else:
            pred_plot = predictions
            target_plot = targets
        
        plt.figure(figsize=(15, 10))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(target_plot, label='Actual', alpha=0.7)
        plt.plot(pred_plot, label='Predicted', alpha=0.7)
        plt.title(f'{title} - Time Series')
        plt.xlabel('Time Step')
        plt.ylabel('CO2 (ppm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(target_plot, pred_plot, alpha=0.6)
        min_val = min(target_plot.min(), pred_plot.min())
        max_val = max(target_plot.max(), pred_plot.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('Actual CO2 (ppm)')
        plt.ylabel('Predicted CO2 (ppm)')
        plt.title('Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 3)
        residuals = target_plot - pred_plot
        plt.scatter(pred_plot, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('Predicted CO2 (ppm)')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Residuals histogram
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_forecast(self, results, original_data=None, save_path=None, title="CO2 Forecast"):
        """Plot forecast with historical data"""
        predictions = results['predictions']
        targets = results['targets']
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data if provided
        if original_data is not None:
            plt.plot(original_data.index, original_data['co2'], 
                    label='Historical Data', alpha=0.7, color='blue')
        
        # Create time index for predictions
        if original_data is not None:
            last_date = original_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=len(predictions), 
                freq='M'
            )
        else:
            forecast_dates = range(len(predictions))
        
        # Plot predictions and targets
        if len(predictions.shape) > 1:
            # Multi-step forecasting
            for i in range(predictions.shape[1]):
                if i == 0:
                    plt.plot(forecast_dates, predictions[:, i], 
                           label=f'Predicted (t+{i+1})', alpha=0.8)
                    plt.plot(forecast_dates, targets[:, i], 
                           label=f'Actual (t+{i+1})', alpha=0.8, linestyle='--')
                else:
                    plt.plot(forecast_dates, predictions[:, i], alpha=0.8)
                    plt.plot(forecast_dates, targets[:, i], alpha=0.8, linestyle='--')
        else:
            plt.plot(forecast_dates, predictions, label='Predicted', alpha=0.8)
            plt.plot(forecast_dates, targets, label='Actual', alpha=0.8, linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('CO2 (ppm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Forecast plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

class MultiModelComparison:
    def __init__(self, models_results, model_names):
        self.models_results = models_results
        self.model_names = model_names
    
    def compare_metrics(self):
        """Compare metrics across models"""
        metrics_df = pd.DataFrame()
        
        for name, results in zip(self.model_names, self.models_results):
            model_metrics = results['metrics']
            metrics_df[name] = pd.Series(model_metrics)
        
        return metrics_df
    
    def plot_comparison(self, save_path=None):
        """Plot model comparison"""
        metrics_df = self.compare_metrics()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in metrics_df.index:
                ax = axes[i]
                metrics_df.loc[metric].plot(kind='bar', ax=ax)
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics_to_plot) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        return metrics_df
    
    def plot_predictions_comparison(self, save_path=None):
        """Plot predictions from all models"""
        plt.figure(figsize=(15, 10))
        
        # Get the first model's targets as reference
        targets = self.models_results[0]['targets']
        if len(targets.shape) > 1:
            targets = targets[:, 0]  # Take first step for multi-step
        
        plt.plot(targets, label='Actual', linewidth=2, color='black')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (name, results) in enumerate(zip(self.model_names, self.models_results)):
            predictions = results['predictions']
            if len(predictions.shape) > 1:
                predictions = predictions[:, 0]  # Take first step for multi-step
            
            color = colors[i % len(colors)]
            plt.plot(predictions, label=f'{name}', alpha=0.7, color=color)
        
        plt.title('Model Predictions Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('CO2 (ppm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions comparison plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()