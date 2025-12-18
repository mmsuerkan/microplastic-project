"""
Ensemble Model Prediction - Kolay kullanım için
Kullanım: python predict.py veya import predict
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Fiziksel sabitler
G = 9.81
RHO_WATER = 1000
NU_WATER = 1.0e-6


class SettlingVelocityNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


class EnsemblePredictor:
    """Ensemble (RF + NN) tahmin sınıfı"""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_dir = model_dir
        self.rf = None
        self.nn = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.features = None
        self.loaded = False

    def load(self):
        """Modelleri yükle"""
        # RF
        rf_path = os.path.join(self.model_dir, 'ensemble_rf.joblib')
        self.rf = joblib.load(rf_path)

        # NN
        nn_path = os.path.join(self.model_dir, 'ensemble_nn.pth')
        checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)

        self.features = checkpoint['features']
        input_size = checkpoint['input_size']

        self.nn = SettlingVelocityNN(input_size)
        self.nn.load_state_dict(checkpoint['model_state'])
        self.nn.eval()

        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_scale = checkpoint['scaler_scale']

        self.loaded = True
        print(f"Model yüklendi: {len(self.features)} feature")

    def add_physics_features(self, df):
        """Gerekli physics feature'ları ekle"""
        df = df.copy()
        d_m = df['d_eq'] / 1000
        delta_rho = df['density'] - RHO_WATER

        df['D_star'] = d_m * ((G * np.abs(delta_rho) / RHO_WATER) / (NU_WATER ** 2)) ** (1/3)
        df['rho_relative'] = df['density'] / RHO_WATER

        mu = NU_WATER * RHO_WATER
        df['v_stokes'] = (G * (d_m ** 2) * np.abs(delta_rho)) / (18 * mu)

        df['flatness'] = df['S'] / df['I']
        df['elongation'] = df['I'] / df['L']

        return df

    def predict(self, df, method='ensemble'):
        """
        Tahmin yap

        Args:
            df: DataFrame veya dict - parçacık özellikleri
            method: 'ensemble', 'rf', veya 'nn'

        Returns:
            velocity (m/s)
        """
        if not self.loaded:
            self.load()

        # Dict ise DataFrame'e çevir
        if isinstance(df, dict):
            df = pd.DataFrame([df])

        # Physics features ekle
        df = self.add_physics_features(df)

        # Feature'ları al
        X = df[self.features].values

        # RF tahmin
        rf_pred = self.rf.predict(X)

        # NN tahmin
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        with torch.no_grad():
            nn_pred = self.nn(torch.FloatTensor(X_scaled)).squeeze().numpy()

        if method == 'rf':
            return rf_pred
        elif method == 'nn':
            return nn_pred
        else:  # ensemble
            return (rf_pred + nn_pred) / 2

    def predict_single(self, a, b, c, density, cat_enc=0, mat_enc=0, view_enc=0):
        """
        Tek parçacık için tahmin (basit arayüz)

        Args:
            a, b, c: Parçacık boyutları (mm)
            density: Yoğunluk (kg/m³)
            cat_enc, mat_enc, view_enc: Kategorik kodlar

        Returns:
            velocity (cm/s)
        """
        # Boyut hesaplamaları
        d_eq = (a * b * c) ** (1/3)
        L = max(a, b, c)
        S = min(a, b, c)
        I = sorted([a, b, c])[1]

        volume = (4/3) * np.pi * (a/2) * (b/2) * (c/2) / 1000**3  # m³
        surface_area = 4 * np.pi * ((((a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6) / 3) ** (1/1.6)) / 4 / 1000**2  # m²

        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        aspect_ratio = L / S if S > 0 else 1
        CSF = S / np.sqrt(L * I) if L * I > 0 else 1

        delta_rho = density - RHO_WATER

        particle = {
            'a': a, 'b': b, 'c': c,
            'd_eq': d_eq, 'L': L, 'I': I, 'S': S,
            'volume': volume, 'surface_area': surface_area,
            'sphericity': sphericity, 'aspect_ratio': aspect_ratio, 'CSF': CSF,
            'density': density, 'delta_rho': delta_rho,
            'cat_enc': cat_enc, 'mat_enc': mat_enc, 'view_enc': view_enc
        }

        velocity_ms = self.predict(particle)[0]
        return velocity_ms * 100  # cm/s


# Global predictor instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
        _predictor.load()
    return _predictor


def predict_velocity(a, b, c, density, cat_enc=0, mat_enc=0, view_enc=0):
    """
    Hızlı tahmin fonksiyonu

    Args:
        a, b, c: Boyutlar (mm)
        density: Yoğunluk (kg/m³)

    Returns:
        velocity (cm/s)
    """
    return get_predictor().predict_single(a, b, c, density, cat_enc, mat_enc, view_enc)


if __name__ == "__main__":
    print("=" * 60)
    print("ENSEMBLE MODEL - Prediction Demo")
    print("=" * 60)

    # Predictor yükle
    predictor = EnsemblePredictor()
    predictor.load()

    # Örnek tahminler
    print("\nÖrnek Tahminler:")
    print("-" * 60)

    examples = [
        {"a": 3.0, "b": 3.0, "c": 3.0, "density": 1050, "desc": "3mm küre, 1050 kg/m³"},
        {"a": 5.0, "b": 5.0, "c": 5.0, "density": 1200, "desc": "5mm küre, 1200 kg/m³"},
        {"a": 4.0, "b": 2.0, "c": 1.0, "density": 1100, "desc": "4x2x1mm elips, 1100 kg/m³"},
        {"a": 2.0, "b": 2.0, "c": 2.0, "density": 900, "desc": "2mm küre, 900 kg/m³ (yüzer)"},
    ]

    for ex in examples:
        v = predictor.predict_single(ex['a'], ex['b'], ex['c'], ex['density'])
        print(f"  {ex['desc']}")
        print(f"    → Tahmin: {v:.2f} cm/s ({v/100:.4f} m/s)")
        print()

    print("=" * 60)
    print("Kullanım:")
    print("  from predict import predict_velocity")
    print("  v = predict_velocity(a=3, b=3, c=3, density=1050)")
    print("=" * 60)
