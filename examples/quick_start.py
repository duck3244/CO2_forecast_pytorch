#!/usr/bin/env python3
"""
Quick Start Example for CO2 Forecasting

This script demonstrates how to quickly train and evaluate a CO2 forecasting model.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
from src.data.data_loader import CO2DataLoader
from src.models.models import create_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.visualization import CO2Visualizer


def quick_start_example():
    """Quick start example with minimal configuration"""

    print("🚀 CO2 Forecasting Quick Start Example")
    print("=" * 50)

    # Simple configuration
    config = {
        'data': {
            'url': "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
            'sequence_length': 24,
            'forecast_horizon': 12,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        },
        'preprocessing': {
            'normalize': True,
            'add_seasonal_features': True,
            'add_trend_features': True,
            'add_lag_features': True,
            'lag_periods': [1, 12, 24]
        },
        'models': {
            'lstm': {
                'input_size': 9,  # co2 + month + month_sin + month_cos + trend + trend_normalized + 3 lag features
                'hidden_size': 32,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': False
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'weight_decay': 1e-5,
            'scheduler': "ReduceLROnPlateau",
            'scheduler_patience': 5,
            'scheduler_factor': 0.5
        },
        'paths': {
            'model_dir': 'models',
            'plot_dir': 'plots'
        }
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")

    # 1. Load and prepare data
    print("\n📊 Loading and preparing data...")
    data_loader = CO2DataLoader(config)
    train_loader, val_loader, test_loader, original_data = data_loader.prepare_data()

    # Visualize data
    visualizer = CO2Visualizer(config)
    visualizer.plot_data_overview(original_data)

    # 2. Create and train model
    print("\n🤖 Creating and training LSTM model...")
    model = create_model('lstm', config)
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Train the model
    training_results = trainer.train()

    # Plot training history
    visualizer.plot_training_history(
        training_results['train_losses'],
        training_results['val_losses']
    )

    # 3. Evaluate model
    print("\n📈 Evaluating model...")
    evaluator = ModelEvaluator(model, test_loader, device, data_loader)
    results = evaluator.evaluate()

    # Print metrics
    print("\n📊 Test Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.6f}")

    # Plot predictions
    evaluator.plot_predictions(results, title="LSTM CO2 Predictions")

    # 4. Generate future forecasts
    print("\n🔮 Generating future forecasts...")

    # Use the last sequence from the test set for forecasting
    model.eval()
    with torch.no_grad():
        # Get the last sequence
        last_sequence = None
        for data, _ in test_loader:
            last_sequence = data[-1:].to(device)  # Take last sample

        if last_sequence is not None:
            future_prediction = model(last_sequence)
            future_prediction = future_prediction.cpu().numpy().flatten()

            # Inverse transform if scaler is available
            if hasattr(data_loader, 'feature_scalers') and 'co2' in data_loader.feature_scalers:
                future_prediction = data_loader.inverse_transform_co2(future_prediction)

            print(f"🔮 Next {config['data']['forecast_horizon']} months forecast:")
            for i, pred in enumerate(future_prediction):
                print(f"  Month {i + 1}: {pred:.2f} ppm")

    print("\n✅ Quick start example completed!")
    print("💡 To run the full pipeline with all models, use: python main.py --mode all")


if __name__ == "__main__":
    quick_start_example()