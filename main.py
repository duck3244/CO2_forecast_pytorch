import yaml
import torch
import os
import argparse
from src.data.data_loader import CO2DataLoader
from src.models.models import create_model, create_ensemble
from src.training.trainer import Trainer, EnsembleTrainer
from src.evaluation.evaluator import ModelEvaluator, MultiModelComparison
from src.training.metrics import print_metrics

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_directories(config):
    """Create necessary directories"""
    for path_key in ['model_dir', 'log_dir', 'plot_dir']:
        if path_key in config['paths']:
            os.makedirs(config['paths'][path_key], exist_ok=True)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def train_single_model(model_type, config, train_loader, val_loader, device):
    """Train a single model"""
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*50}")
    
    # Create model
    model = create_model(model_type, config)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Train
    training_results = trainer.train()
    
    # Save model
    model_path = os.path.join(config['paths']['model_dir'], f'{model_type}_model.pth')
    trainer.save_model(model_path)
    
    return model, training_results

def evaluate_model(model, test_loader, data_loader, model_name, config):
    """Evaluate a trained model"""
    print(f"\n{'='*30}")
    print(f"Evaluating {model_name}")
    print(f"{'='*30}")
    
    device = next(model.parameters()).device
    evaluator = ModelEvaluator(model, test_loader, device, data_loader)
    results = evaluator.evaluate()
    
    print_metrics(results['metrics'], f"{model_name} Test Metrics")
    
    # Plot results
    plot_path = os.path.join(config['paths']['plot_dir'], f'{model_name}_predictions.png')
    evaluator.plot_predictions(results, save_path=plot_path, title=f"{model_name} Predictions")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='CO2 Forecasting with PyTorch')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'all'], 
                        default='all', help='Mode to run')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'hybrid', 'ensemble'], 
                        default='all', help='Model type to train/evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    setup_directories(config)
    
    # Get device
    device = get_device()
    
    # Prepare data
    print("Preparing data...")
    data_loader = CO2DataLoader(config)
    train_loader, val_loader, test_loader, original_data = data_loader.prepare_data()
    
    if args.mode in ['train', 'all']:
        trained_models = {}
        training_results = {}
        
        if args.model == 'all' or args.model == 'lstm':
            model, results = train_single_model('lstm', config, train_loader, val_loader, device)
            trained_models['lstm'] = model
            training_results['lstm'] = results
        
        if args.model == 'all' or args.model == 'transformer':
            model, results = train_single_model('transformer', config, train_loader, val_loader, device)
            trained_models['transformer'] = model
            training_results['transformer'] = results
        
        if args.model == 'all' or args.model == 'hybrid':
            model, results = train_single_model('hybrid', config, train_loader, val_loader, device)
            trained_models['hybrid'] = model
            training_results['hybrid'] = results
        
        if args.model == 'all' or args.model == 'ensemble':
            print(f"\n{'='*50}")
            print("Training Ensemble Model")
            print(f"{'='*50}")
            
            # Create individual models for ensemble
            individual_models = []
            for model_type in config['ensemble']['models']:
                if model_type in trained_models:
                    individual_models.append(trained_models[model_type])
                else:
                    individual_models.append(create_model(model_type, config))
            
            # Train ensemble
            ensemble_trainer = EnsembleTrainer(individual_models, train_loader, val_loader, config, device)
            ensemble_trainer.fine_tune_ensemble()
            
            # Save ensemble model
            ensemble_path = os.path.join(config['paths']['model_dir'], 'ensemble_model.pth')
            torch.save({
                'model_state_dict': ensemble_trainer.ensemble_model.state_dict(),
                'config': config
            }, ensemble_path)
            
            trained_models['ensemble'] = ensemble_trainer.ensemble_model
    
    if args.mode in ['evaluate', 'all']:
        # Load models if not trained in this run
        if args.mode == 'evaluate':
            trained_models = {}
            model_types = ['lstm', 'transformer', 'hybrid', 'ensemble'] if args.model == 'all' else [args.model]
            
            for model_type in model_types:
                model_path = os.path.join(config['paths']['model_dir'], f'{model_type}_model.pth')
                if os.path.exists(model_path):
                    if model_type == 'ensemble':
                        ensemble_model = create_ensemble(config)
                        checkpoint = torch.load(model_path, map_location=device)
                        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
                        trained_models[model_type] = ensemble_model
                    else:
                        model = create_model(model_type, config)
                        checkpoint = torch.load(model_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        trained_models[model_type] = model.to(device)
                else:
                    print(f"Model file not found: {model_path}")
        
        # Evaluate all models
        evaluation_results = {}
        for model_name, model in trained_models.items():
            results = evaluate_model(model, test_loader, data_loader, model_name, config)
            evaluation_results[model_name] = results
        
        # Compare models if multiple models are evaluated
        if len(evaluation_results) > 1:
            print(f"\n{'='*50}")
            print("Model Comparison")
            print(f"{'='*50}")
            
            model_names = list(evaluation_results.keys())
            model_results = list(evaluation_results.values())
            
            comparison = MultiModelComparison(model_results, model_names)
            
            # Print comparison table
            metrics_df = comparison.compare_metrics()
            print("\nMetrics Comparison:")
            print(metrics_df.round(6))
            
            # Plot comparisons
            comparison_plot_path = os.path.join(config['paths']['plot_dir'], 'model_comparison.png')
            comparison.plot_comparison(save_path=comparison_plot_path)
            
            predictions_plot_path = os.path.join(config['paths']['plot_dir'], 'predictions_comparison.png')
            comparison.plot_predictions_comparison(save_path=predictions_plot_path)
            
            # Find best model
            best_model = metrics_df.loc['R2'].idxmax()
            print(f"\nBest model based on R2 score: {best_model}")
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()