import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CO2Visualizer:
    def __init__(self, config=None):
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        """Plot training and validation loss history"""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Training Loss', color=self.colors[0])
        plt.plot(epochs, val_losses, label='Validation Loss', color=self.colors[1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Smoothed loss plot
        plt.subplot(1, 2, 2)
        window = max(1, len(train_losses) // 20)
        smooth_train = pd.Series(train_losses).rolling(window=window).mean()
        smooth_val = pd.Series(val_losses).rolling(window=window).mean()
        
        plt.plot(epochs, smooth_train, label=f'Smoothed Training (w={window})', color=self.colors[0])
        plt.plot(epochs, smooth_val, label=f'Smoothed Validation (w={window})', color=self.colors[1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_data_overview(self, data, save_path=None):
        """Plot data overview and statistics"""
        plt.figure(figsize=(15, 10))
        
        # Original time series
        plt.subplot(2, 3, 1)
        plt.plot(data.index, data['co2'], alpha=0.8)
        plt.title('CO2 Time Series')
        plt.xlabel('Date')
        plt.ylabel('CO2 (ppm)')
        plt.grid(True, alpha=0.3)
        
        # Seasonal decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(data['co2'], model='additive', period=12)
        
        plt.subplot(2, 3, 2)
        plt.plot(decomposition.trend, alpha=0.8)
        plt.title('Trend Component')
        plt.xlabel('Date')
        plt.ylabel('CO2 (ppm)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(decomposition.seasonal, alpha=0.8)
        plt.title('Seasonal Component')
        plt.xlabel('Date')
        plt.ylabel('CO2 (ppm)')
        plt.grid(True, alpha=0.3)
        
        # Distribution
        plt.subplot(2, 3, 4)
        plt.hist(data['co2'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('CO2 Distribution')
        plt.xlabel('CO2 (ppm)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Boxplot by month
        plt.subplot(2, 3, 5)
        monthly_data = data.copy()
        monthly_data['month'] = monthly_data.index.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        box_data = [monthly_data[monthly_data['month'] == i]['co2'].values 
                    for i in range(1, 13)]
        plt.boxplot(box_data, labels=month_names)
        plt.title('Seasonal Variation')
        plt.xlabel('Month')
        plt.ylabel('CO2 (ppm)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Growth rate
        plt.subplot(2, 3, 6)
        growth_rate = data['co2'].pct_change() * 100
        plt.plot(data.index, growth_rate, alpha=0.8)
        plt.title('Growth Rate (%)')
        plt.xlabel('Date')
        plt.ylabel('Growth Rate (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Data overview plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names, importances, save_path=None):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], 
                color=self.colors[0], alpha=0.8)
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_interactive_forecast(self, historical_data, predictions, targets, 
                                forecast_dates, save_path=None):
        """Create interactive forecast plot using Plotly"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CO2 Forecast', 'Prediction Error'),
            vertical_spacing=0.1
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['co2'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Actual values (if available)
        if targets is not None:
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=targets,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='green', width=2, dash='dash'),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Error plot
            errors = targets - predictions
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=errors,
                    mode='lines+markers',
                    name='Prediction Error',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", 
                         opacity=0.5, row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="CO2 (ppm)", row=1, col=1)
        fig.update_yaxes(title_text="Error (ppm)", row=2, col=1)
        
        fig.update_layout(
            title="Interactive CO2 Forecast",
            height=600,
            hovermode='x unified'
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"Interactive forecast plot saved to {save_path}")
        
        fig.show()
    
    def plot_model_architecture(self, model, save_path=None):
        """Visualize model architecture (simplified representation)"""
        plt.figure(figsize=(12, 8))
        
        # This is a simplified visualization
        # For more detailed architecture plots, consider using tools like torchviz
        
        model_type = type(model).__name__
        
        if 'LSTM' in model_type:
            self._plot_lstm_architecture()
        elif 'Transformer' in model_type:
            self._plot_transformer_architecture()
        elif 'Hybrid' in model_type:
            self._plot_hybrid_architecture()
        elif 'Ensemble' in model_type:
            self._plot_ensemble_architecture()
        
        plt.title(f'{model_type} Architecture')
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model architecture plot saved to {save_path}")
        
        plt.show()
    
    def _plot_lstm_architecture(self):
        """Plot LSTM architecture diagram"""
        # Simple block diagram representation
        blocks = ['Input', 'LSTM Layer 1', 'LSTM Layer 2', 'Dense', 'Output']
        positions = [(i, 0) for i in range(len(blocks))]
        
        for i, (pos, block) in enumerate(zip(positions, blocks)):
            rect = plt.Rectangle((pos[0]-0.4, pos[1]-0.2), 0.8, 0.4, 
                               fill=True, facecolor=self.colors[i % len(self.colors)], 
                               alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(pos[0], pos[1], block, ha='center', va='center', fontweight='bold')
            
            if i < len(blocks) - 1:
                plt.arrow(pos[0]+0.4, pos[1], 0.2, 0, head_width=0.05, 
                         head_length=0.05, fc='black', ec='black')
        
        plt.xlim(-0.5, len(blocks)-0.5)
        plt.ylim(-0.5, 0.5)
    
    def _plot_transformer_architecture(self):
        """Plot Transformer architecture diagram"""
        blocks = ['Input', 'Pos Encoding', 'Multi-Head Attention', 'Feed Forward', 'Output']
        positions = [(i, 0) for i in range(len(blocks))]
        
        for i, (pos, block) in enumerate(zip(positions, blocks)):
            rect = plt.Rectangle((pos[0]-0.4, pos[1]-0.2), 0.8, 0.4, 
                               fill=True, facecolor=self.colors[i % len(self.colors)], 
                               alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(pos[0], pos[1], block, ha='center', va='center', fontweight='bold')
            
            if i < len(blocks) - 1:
                plt.arrow(pos[0]+0.4, pos[1], 0.2, 0, head_width=0.05, 
                         head_length=0.05, fc='black', ec='black')
        
        plt.xlim(-0.5, len(blocks)-0.5)
        plt.ylim(-0.5, 0.5)
    
    def _plot_hybrid_architecture(self):
        """Plot Hybrid model architecture diagram"""
        # LSTM branch
        lstm_blocks = ['Input', 'LSTM', 'Features']
        lstm_positions = [(i, 0.3) for i in range(len(lstm_blocks))]
        
        # Transformer branch
        trans_blocks = ['Input', 'Transformer', 'Features']
        trans_positions = [(i, -0.3) for i in range(len(trans_blocks))]
        
        # Fusion
        fusion_pos = (3, 0)
        output_pos = (4, 0)
        
        # Draw LSTM branch
        for i, (pos, block) in enumerate(zip(lstm_positions, lstm_blocks)):
            rect = plt.Rectangle((pos[0]-0.3, pos[1]-0.1), 0.6, 0.2, 
                               fill=True, facecolor=self.colors[0], 
                               alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(pos[0], pos[1], block, ha='center', va='center', fontweight='bold')
            
            if i < len(lstm_blocks) - 1:
                plt.arrow(pos[0]+0.3, pos[1], 0.4, 0, head_width=0.03, 
                         head_length=0.03, fc='black', ec='black')
        
        # Draw Transformer branch
        for i, (pos, block) in enumerate(zip(trans_positions, trans_blocks)):
            rect = plt.Rectangle((pos[0]-0.3, pos[1]-0.1), 0.6, 0.2, 
                               fill=True, facecolor=self.colors[1], 
                               alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(pos[0], pos[1], block, ha='center', va='center', fontweight='bold')
            
            if i < len(trans_blocks) - 1:
                plt.arrow(pos[0]+0.3, pos[1], 0.4, 0, head_width=0.03, 
                         head_length=0.03, fc='black', ec='black')
        
        # Fusion layer
        rect = plt.Rectangle((fusion_pos[0]-0.3, fusion_pos[1]-0.1), 0.6, 0.2, 
                           fill=True, facecolor=self.colors[2], 
                           alpha=0.7, edgecolor='black')
        plt.gca().add_patch(rect)
        plt.text(fusion_pos[0], fusion_pos[1], 'Fusion', ha='center', va='center', fontweight='bold')
        
        # Output
        rect = plt.Rectangle((output_pos[0]-0.3, output_pos[1]-0.1), 0.6, 0.2, 
                           fill=True, facecolor=self.colors[3], 
                           alpha=0.7, edgecolor='black')
        plt.gca().add_patch(rect)
        plt.text(output_pos[0], output_pos[1], 'Output', ha='center', va='center', fontweight='bold')
        
        # Arrows to fusion
        plt.arrow(2.3, 0.3, 0.4, -0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        plt.arrow(2.3, -0.3, 0.4, 0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        plt.arrow(3.3, 0, 0.4, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        plt.xlim(-0.5, 4.5)
        plt.ylim(-0.5, 0.5)
    
    def _plot_ensemble_architecture(self):
        """Plot Ensemble model architecture diagram"""
        models = ['LSTM', 'Transformer', 'Hybrid']
        model_positions = [(i, 0.5) for i in range(len(models))]
        ensemble_pos = (1, -0.5)
        output_pos = (1, -1)
        
        # Draw individual models
        for i, (pos, model) in enumerate(zip(model_positions, models)):
            rect = plt.Rectangle((pos[0]-0.3, pos[1]-0.1), 0.6, 0.2, 
                               fill=True, facecolor=self.colors[i], 
                               alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(pos[0], pos[1], model, ha='center', va='center', fontweight='bold')
            
            # Arrow to ensemble
            plt.arrow(pos[0], pos[1]-0.1, 0, -0.2, head_width=0.05, 
                     head_length=0.05, fc='black', ec='black')
        
        # Ensemble layer
        rect = plt.Rectangle((ensemble_pos[0]-0.4, ensemble_pos[1]-0.1), 0.8, 0.2, 
                           fill=True, facecolor=self.colors[3], 
                           alpha=0.7, edgecolor='black')
        plt.gca().add_patch(rect)
        plt.text(ensemble_pos[0], ensemble_pos[1], 'Weighted Avg', ha='center', va='center', fontweight='bold')
        
        # Output
        rect = plt.Rectangle((output_pos[0]-0.3, output_pos[1]-0.1), 0.6, 0.2, 
                           fill=True, facecolor=self.colors[4], 
                           alpha=0.7, edgecolor='black')
        plt.gca().add_patch(rect)
        plt.text(output_pos[0], output_pos[1], 'Output', ha='center', va='center', fontweight='bold')
        
        # Arrow to output
        plt.arrow(ensemble_pos[0], ensemble_pos[1]-0.1, 0, -0.3, head_width=0.05, 
                 head_length=0.05, fc='black', ec='black')
        
        plt.xlim(-0.5, 2.5)
        plt.ylim(-1.2, 0.7)