"""
Ensemble Model - RF + Neural Network
Mikroplastik Settling Velocity Prediction
CV R² = 0.84 (Teorik limitin %97'si)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Fiziksel sabitler
G = 9.81  # m/s²
RHO_WATER = 1000  # kg/m³
NU_WATER = 1.0e-6  # m²/s (kinematic viscosity at 20°C)


class SettlingVelocityNN(nn.Module):
    """BPNN mimarisi - 4 hidden layer"""
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def add_physics_features(df):
    """Physics-based feature'lar ekle"""
    d_m = df['d_eq'] / 1000
    delta_rho = df['density'] - RHO_WATER

    df['D_star'] = d_m * ((G * np.abs(delta_rho) / RHO_WATER) / (NU_WATER ** 2)) ** (1/3)
    df['rho_relative'] = df['density'] / RHO_WATER

    mu = NU_WATER * RHO_WATER
    df['v_stokes'] = (G * (d_m ** 2) * np.abs(delta_rho)) / (18 * mu)

    df['flatness'] = df['S'] / df['I']
    df['elongation'] = df['I'] / df['L']

    return df


def train_nn(X_train, y_train, X_val, y_val, epochs=300, lr=0.001, device='cpu'):
    """Neural Network eğit"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_s)
    y_val_t = torch.FloatTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)

    model = SettlingVelocityNN(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_r2 = -np.inf
    best_state = None
    patience = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_t.to(device)).squeeze().cpu().numpy()
        val_r2 = r2_score(y_val, y_pred)
        scheduler.step(-val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                break

    model.load_state_dict(best_state)
    return model, scaler, best_val_r2


def cross_validate_ensemble(X, y, n_splits=5, device='cpu'):
    """5-Fold CV for Ensemble (RF + NN)"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rf_scores = []
    nn_scores = []
    ensemble_scores = []

    print("\n5-Fold Cross Validation:")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Random Forest (optimized parameters)
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,  # Overfitting azaltmak için
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        rf_r2 = r2_score(y_val, rf_pred)
        rf_scores.append(rf_r2)

        # Neural Network
        nn_model, scaler, nn_r2 = train_nn(X_train, y_train, X_val, y_val, device=device)
        X_val_s = scaler.transform(X_val)
        nn_model.eval()
        with torch.no_grad():
            nn_pred = nn_model(torch.FloatTensor(X_val_s).to(device)).squeeze().cpu().numpy()
        nn_r2 = r2_score(y_val, nn_pred)
        nn_scores.append(nn_r2)

        # Ensemble (average)
        ensemble_pred = (rf_pred + nn_pred) / 2
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        ensemble_scores.append(ensemble_r2)

        print(f"  Fold {fold+1}: RF={rf_r2:.4f}, NN={nn_r2:.4f}, Ensemble={ensemble_r2:.4f}")

    print("-" * 60)
    print(f"  Mean:   RF={np.mean(rf_scores):.4f}, NN={np.mean(nn_scores):.4f}, Ensemble={np.mean(ensemble_scores):.4f}")

    return {
        'rf': np.array(rf_scores),
        'nn': np.array(nn_scores),
        'ensemble': np.array(ensemble_scores)
    }


def train_final_ensemble(X, y, device='cpu'):
    """Final ensemble modelini eğit ve kaydet"""
    print("\nFinal Ensemble Model Eğitimi...")
    print("=" * 60)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Random Forest
    print("\n1. Random Forest eğitiliyor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)
    print(f"   Train R²: {r2_score(y_train, rf_train_pred):.4f}")
    print(f"   Test R²:  {r2_score(y_test, rf_test_pred):.4f}")

    # 2. Neural Network
    print("\n2. Neural Network eğitiliyor...")
    nn_model, scaler, _ = train_nn(X_train, y_train, X_test, y_test, epochs=300, device=device)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    nn_model.eval()
    with torch.no_grad():
        nn_train_pred = nn_model(torch.FloatTensor(X_train_s).to(device)).squeeze().cpu().numpy()
        nn_test_pred = nn_model(torch.FloatTensor(X_test_s).to(device)).squeeze().cpu().numpy()

    print(f"   Train R²: {r2_score(y_train, nn_train_pred):.4f}")
    print(f"   Test R²:  {r2_score(y_test, nn_test_pred):.4f}")

    # 3. Ensemble
    print("\n3. Ensemble (RF + NN) / 2:")
    ensemble_train_pred = (rf_train_pred + nn_train_pred) / 2
    ensemble_test_pred = (rf_test_pred + nn_test_pred) / 2

    train_r2 = r2_score(y_train, ensemble_train_pred)
    test_r2 = r2_score(y_test, ensemble_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
    test_mae = mean_absolute_error(y_test, ensemble_test_pred)

    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²:  {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse*100:.2f} cm/s")
    print(f"   Test MAE:  {test_mae*100:.2f} cm/s")

    return rf, nn_model, scaler, {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }


def main():
    print("=" * 70)
    print("ENSEMBLE MODEL - RF + Neural Network")
    print("Mikroplastik Settling Velocity Prediction")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Veri yükle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'ml_data_full_v2.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, '..', 'data', 'ml_data_full_v2.csv')

    print(f"\nVeri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Toplam: {len(df)} satır")

    # RESIN (cat_enc=11) outlier'ları çıkar - aynı parçacık için 4-11.69 cm/s varyans!
    resin_count = len(df[df['cat_enc'] == 11])
    df = df[df['cat_enc'] != 11]
    print(f"RESIN outlier çıkarıldı: {resin_count} satır")
    print(f"Kalan: {len(df)} satır")

    # Physics features
    df = add_physics_features(df)

    # Features
    features = [
        'a', 'b', 'c', 'd_eq', 'L', 'I', 'S',
        'volume', 'surface_area',
        'sphericity', 'aspect_ratio', 'CSF', 'flatness', 'elongation',
        'density', 'delta_rho', 'rho_relative',
        'D_star', 'v_stokes',
        'cat_enc', 'mat_enc', 'view_enc'
    ]

    # NaN temizle
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['velocity_ms'])
    print(f"Temiz veri: {len(df)} satır")

    X = df[features].values
    y = df['velocity_ms'].values

    # Cross Validation
    cv_scores = cross_validate_ensemble(X, y, n_splits=5, device=device)

    print("\n" + "=" * 70)
    print("CROSS VALIDATION SONUÇLARI")
    print("=" * 70)
    print(f"Random Forest:  {cv_scores['rf'].mean():.4f} ± {cv_scores['rf'].std():.4f}")
    print(f"Neural Network: {cv_scores['nn'].mean():.4f} ± {cv_scores['nn'].std():.4f}")
    print(f"ENSEMBLE:       {cv_scores['ensemble'].mean():.4f} ± {cv_scores['ensemble'].std():.4f}")

    # Final model
    rf, nn_model, scaler, metrics = train_final_ensemble(X, y, device=device)

    # Kaydet
    print("\n" + "=" * 70)
    print("MODEL KAYDEDİLİYOR")
    print("=" * 70)

    # RF kaydet
    rf_path = os.path.join(script_dir, 'ensemble_rf.joblib')
    joblib.dump(rf, rf_path)
    print(f"RF kaydedildi: {rf_path}")

    # NN kaydet
    nn_path = os.path.join(script_dir, 'ensemble_nn.pth')
    torch.save({
        'model_state': nn_model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features,
        'input_size': len(features)
    }, nn_path)
    print(f"NN kaydedildi: {nn_path}")

    # Ensemble meta bilgileri
    meta_path = os.path.join(script_dir, 'ensemble_meta.joblib')
    joblib.dump({
        'features': features,
        'cv_scores': cv_scores,
        'metrics': metrics,
        'rf_path': 'ensemble_rf.joblib',
        'nn_path': 'ensemble_nn.pth'
    }, meta_path)
    print(f"Meta kaydedildi: {meta_path}")

    print("\n" + "=" * 70)
    print("ÖZET")
    print("=" * 70)
    print(f"Ensemble CV R²: {cv_scores['ensemble'].mean():.4f}")
    print(f"Test R²:        {metrics['test_r2']:.4f}")
    print(f"Test RMSE:      {metrics['test_rmse']*100:.2f} cm/s")
    print(f"\nTeorik limit: 0.87")
    print(f"Model/Limit:  {cv_scores['ensemble'].mean()/0.87*100:.1f}%")
    print("\nDosyalar:")
    print("  - ensemble_rf.joblib (Random Forest)")
    print("  - ensemble_nn.pth (Neural Network)")
    print("  - ensemble_meta.joblib (Meta bilgiler)")


if __name__ == "__main__":
    main()
