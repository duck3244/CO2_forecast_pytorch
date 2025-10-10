import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CO2LSTM(nn.Module):
    """LSTM-based CO2 forecasting model"""
    
    def __init__(self, config):
        super(CO2LSTM, self).__init__()
        self.config = config
        model_config = config['models']['lstm']
        
        self.input_size = model_config['input_size']
        self.hidden_size = model_config['hidden_size']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        self.forecast_horizon = config['data']['forecast_horizon']
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=model_config.get('bidirectional', False)
        )
        
        lstm_output_size = self.hidden_size * (2 if model_config.get('bidirectional', False) else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.forecast_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = self.fc(lstm_out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CO2Transformer(nn.Module):
    """Transformer-based CO2 forecasting model"""
    
    def __init__(self, config):
        super(CO2Transformer, self).__init__()
        self.config = config
        model_config = config['models']['transformer']
        
        self.input_size = config['models']['lstm']['input_size']  # Same input size
        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        self.forecast_horizon = config['data']['forecast_horizon']
        
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.forecast_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.input_projection(x)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        transformer_out = self.transformer(x)
        # Take the last output
        out = self.fc(transformer_out[:, -1, :])
        return out

class CO2HybridModel(nn.Module):
    """Hybrid LSTM-Transformer model"""
    
    def __init__(self, config):
        super(CO2HybridModel, self).__init__()
        self.config = config
        model_config = config['models']['hybrid']
        
        self.input_size = config['models']['lstm']['input_size']
        self.lstm_hidden = model_config['lstm_hidden']
        self.transformer_d_model = model_config['transformer_d_model']
        self.fusion_hidden = model_config['fusion_hidden']
        self.forecast_horizon = config['data']['forecast_horizon']
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Transformer branch
        self.input_projection = nn.Linear(self.input_size, self.transformer_d_model)
        self.pos_encoding = PositionalEncoding(self.transformer_d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_d_model,
            nhead=4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, model_config['transformer_layers'])
        
        # Fusion layer
        combined_size = self.lstm_hidden + self.transformer_d_model
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_hidden, self.fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_hidden // 2, self.forecast_horizon)
        )
    
    def forward(self, x):
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]  # Last output
        
        # Transformer branch
        transformer_input = self.input_projection(x)
        transformer_input = transformer_input.transpose(0, 1)
        transformer_input = self.pos_encoding(transformer_input)
        transformer_input = transformer_input.transpose(0, 1)
        
        transformer_out = self.transformer(transformer_input)
        transformer_features = transformer_out[:, -1, :]  # Last output
        
        # Fusion
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        out = self.fusion(combined)
        
        return out

class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, forecast_horizon)
        weights = F.softmax(self.weights, dim=0)
        weighted_output = torch.sum(stacked_outputs * weights.view(-1, 1, 1), dim=0)
        
        return weighted_output

def create_model(model_type, config):
    """Factory function to create models"""
    if model_type == 'lstm':
        return CO2LSTM(config)
    elif model_type == 'transformer':
        return CO2Transformer(config)
    elif model_type == 'hybrid':
        return CO2HybridModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_ensemble(config):
    """Create ensemble model"""
    model_types = config['ensemble']['models']
    models = [create_model(model_type, config) for model_type in model_types]
    
    weights = config['ensemble'].get('weights', None)
    ensemble = EnsembleModel(models, weights)
    
    return ensemble