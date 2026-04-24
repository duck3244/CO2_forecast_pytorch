import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler
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
        self.scaler = StandardScaler()
        self.feature_scalers = {}
        self.raw_data = None
        self.processed_data = None
        self.detrend_enabled = False
        self.yoy_diff_enabled = False
        self.trend_slope = 0.0
        self.trend_intercept = 0.0
        self.trend_series = None  # pd.Series indexed by date; trend value per timestamp
        self._target_dates = {}   # split_name -> list of DatetimeIndex (one per sequence)
        
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
    
    def _apply_yoy_diff(self, df):
        """Replace co2 with 12-month YoY difference. Preserves raw in self.raw_data."""
        df = df.copy()
        df['co2'] = df['co2'] - df['co2'].shift(12)
        return df

    def _fit_and_apply_detrend(self, df):
        """Fit polynomial trend on training portion of raw co2, subtract across full df."""
        df = df.copy()
        n = len(df)
        t = np.arange(n)
        train_end = int(n * self.config['data']['train_ratio'])
        degree = int(self.config['preprocessing'].get('trend_degree', 1))
        coeffs = np.polyfit(t[:train_end], df['co2'].iloc[:train_end].values, degree)
        self.trend_coeffs = coeffs
        self.trend_degree = degree
        self.trend_series = pd.Series(np.polyval(coeffs, t), index=df.index)
        if degree == 1:
            self.trend_slope = float(coeffs[0])
            self.trend_intercept = float(coeffs[1])
        df['co2'] = df['co2'].values - self.trend_series.values
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
                self.feature_scalers[col] = StandardScaler()
            
            if fit:
                df_normalized[col] = self.feature_scalers[col].fit_transform(df[[col]])
            else:
                df_normalized[col] = self.feature_scalers[col].transform(df[[col]])
        
        return df_normalized
    
    def create_sequences(self, data, sequence_length, forecast_horizon):
        """Create sequences for training. Returns sequences, targets, and DatetimeIndex per target."""
        sequences = []
        targets = []
        target_dates = []

        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            seq = data.iloc[i:i + sequence_length].values
            tgt_slice = slice(i + sequence_length, i + sequence_length + forecast_horizon)
            target = data['co2'].iloc[tgt_slice].values
            sequences.append(seq)
            targets.append(target)
            target_dates.append(data.index[tgt_slice])

        return np.array(sequences), np.array(targets), target_dates
    
    def prepare_data(self):
        """Main data preparation pipeline"""
        if self.raw_data is None:
            self.download_data()

        df = self.raw_data

        # Optional detrending or YoY differencing (mutually exclusive)
        self.detrend_enabled = self.config['preprocessing'].get('detrend', False)
        self.yoy_diff_enabled = self.config['preprocessing'].get('yoy_diff', False)
        if self.detrend_enabled and self.yoy_diff_enabled:
            raise ValueError("preprocessing.detrend and preprocessing.yoy_diff are mutually exclusive")
        if self.detrend_enabled:
            df = self._fit_and_apply_detrend(df)
        elif self.yoy_diff_enabled:
            df = self._apply_yoy_diff(df)

        # Create features (lag, seasonal, trend index) on (possibly detrended) co2
        df_features = self.create_features(df)

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

        train_seq, train_targets, train_dates = self.create_sequences(train_data, seq_len, horizon)
        val_seq, val_targets, val_dates = self.create_sequences(val_data, seq_len, horizon)
        test_seq, test_targets, test_dates = self.create_sequences(test_data, seq_len, horizon)

        self._target_dates = {'train': train_dates, 'val': val_dates, 'test': test_dates}

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
    
    def inverse_transform_co2(self, scaled_values, split=None):
        """Inverse transform CO2 values. Undoes scaler and detrending if active.

        Args:
            scaled_values: np.ndarray of shape (n_sequences, horizon) or 1D.
            split: one of 'train'/'val'/'test' — required when detrend is enabled to align
                   per-sequence trend values with target dates.
        """
        values = np.asarray(scaled_values)
        shape = values.shape

        # 1) undo scaler
        if 'co2' in self.feature_scalers:
            flat = self.feature_scalers['co2'].inverse_transform(values.reshape(-1, 1))
            values = flat.reshape(shape)

        # 2a) add linear/polynomial trend back
        if self.detrend_enabled and split is not None and split in self._target_dates:
            dates = self._target_dates[split]
            if len(dates) and values.ndim == 2 and values.shape[0] == len(dates):
                trend_matrix = np.stack(
                    [self.trend_series.reindex(d).values for d in dates], axis=0
                )
                values = values + trend_matrix
        # 2b) YoY inverse: predicted_diff + raw_co2[t - 12 months]
        elif self.yoy_diff_enabled and split is not None and split in self._target_dates:
            dates = self._target_dates[split]
            if len(dates) and values.ndim == 2 and values.shape[0] == len(dates):
                prev_year_matrix = np.stack(
                    [self.raw_data['co2'].reindex(d - pd.DateOffset(months=12)).values
                     for d in dates],
                    axis=0,
                )
                values = values + prev_year_matrix

        # Preserve legacy 1D-flatten behavior for callers that pass 1D
        if np.asarray(scaled_values).ndim == 1:
            return values.flatten()
        return values