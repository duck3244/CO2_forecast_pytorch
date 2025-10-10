import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class CO2Dataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class CO2DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.raw_data = None
        self.processed_data = None
        
    def download_data(self):
        """Download CO2 data from NOAA"""
        print("Downloading CO2 data from NOAA...")
        
        try:
            response = requests.get(self.config['data']['url'])
            response.raise_for_status()
            
            # Parse the text data
            lines = response.text.strip().split('\n')
            data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            
            # Create DataFrame
            data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 4:
                    year = int(parts[0])
                    month = int(parts[1])
                    co2 = float(parts[3]) if parts[3] != '-99.99' else np.nan
                    data.append([year, month, co2])
            
            df = pd.DataFrame(data, columns=['year', 'month', 'co2'])
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            df = df.set_index('date')
            df = df[['co2']].dropna()
            
            print(f"Downloaded {len(df)} records from {df.index.min()} to {df.index.max()}")
            self.raw_data = df
            return df
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Using sample data instead...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample CO2 data for testing"""
        dates = pd.date_range('1958-03', '2023-12', freq='M')
        
        # Generate realistic CO2 data with trend and seasonality
        trend = np.linspace(315, 420, len(dates))
        seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 0.5, len(dates))
        co2 = trend + seasonal + noise
        
        df = pd.DataFrame({'co2': co2}, index=dates)
        self.raw_data = df
        return df
    
    def create_features(self, df):
        """Create additional features"""
        df = df.copy()
        
        if self.config['preprocessing']['add_seasonal_features']:
            df['month'] = df.index.month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if self.config['preprocessing']['add_trend_features']:
            df['trend'] = range(len(df))
            df['trend_normalized'] = (df['trend'] - df['trend'].min()) / (df['trend'].max() - df['trend'].min())
        
        if self.config['preprocessing']['add_lag_features']:
            for lag in self.config['preprocessing']['lag_periods']:
                df[f'co2_lag_{lag}'] = df['co2'].shift(lag)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def normalize_features(self, df, fit=True):
        """Normalize features"""
        df_normalized = df.copy()
        
        for col in df.columns:
            if col not in self.feature_scalers:
                self.feature_scalers[col] = MinMaxScaler()
            
            if fit:
                df_normalized[col] = self.feature_scalers[col].fit_transform(df[[col]])
            else:
                df_normalized[col] = self.feature_scalers[col].transform(df[[col]])
        
        return df_normalized
    
    def create_sequences(self, data, sequence_length, forecast_horizon):
        """Create sequences for training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            seq = data.iloc[i:i + sequence_length].values
            target = data['co2'].iloc[i + sequence_length:i + sequence_length + forecast_horizon].values
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self):
        """Main data preparation pipeline"""
        if self.raw_data is None:
            self.download_data()
        
        # Create features
        df_features = self.create_features(self.raw_data)
        
        # Split data
        train_size = int(len(df_features) * self.config['data']['train_ratio'])
        val_size = int(len(df_features) * self.config['data']['val_ratio'])
        
        train_data = df_features.iloc[:train_size]
        val_data = df_features.iloc[train_size:train_size + val_size]
        test_data = df_features.iloc[train_size + val_size:]
        
        # Normalize features
        if self.config['preprocessing']['normalize']:
            train_data = self.normalize_features(train_data, fit=True)
            val_data = self.normalize_features(val_data, fit=False)
            test_data = self.normalize_features(test_data, fit=False)
        
        # Create sequences
        seq_len = self.config['data']['sequence_length']
        horizon = self.config['data']['forecast_horizon']
        
        train_seq, train_targets = self.create_sequences(train_data, seq_len, horizon)
        val_seq, val_targets = self.create_sequences(val_data, seq_len, horizon)
        test_seq, test_targets = self.create_sequences(test_data, seq_len, horizon)
        
        # Create datasets
        train_dataset = CO2Dataset(train_seq, train_targets)
        val_dataset = CO2Dataset(val_seq, val_targets)
        test_dataset = CO2Dataset(test_seq, test_targets)
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, df_features
    
    def inverse_transform_co2(self, scaled_values):
        """Inverse transform CO2 values"""
        if 'co2' in self.feature_scalers:
            return self.feature_scalers['co2'].inverse_transform(scaled_values.reshape(-1, 1)).flatten()
        return scaled_values