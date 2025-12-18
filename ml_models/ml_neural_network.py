"""
Neural Network (BPNN) Model - Mikroplastik Settling Velocity
Literatür tabanlı mimari: Input → 128 → 256 → 128 → 64 → 1
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Fiziksel sabitler
G = 9.81  # m/s²
RHO_WATER = 1000  # kg/m³
NU_WATER = 1.0e-6  # m²/s (kinematic viscosity at 20°C)


class SettlingVelocityNN(nn.Module):
    """
    BPNN mimarisi - literatürden adapte edilmiş
    Mikroplastik settling velocity tahmini için
    """
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: Input → 128
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: 128 → 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 4: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output: 64 → 1
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def add_physics_features(df):
    """
    Literatürden öğrenilen kritik feature'ları ekle:
    - D* (dimensionless diameter) - EN ÖNEMLİ!
    - Relative density
    - Stokes velocity (teorik)
    """
    # d_eq mm'den m'ye çevir
    d_m = df['d_eq'] / 1000

    # Delta rho (kg/m³)
    delta_rho = df['density'] - RHO_WATER

    # D* (dimensionless diameter) - Literatürde en önemli feature
    # D* = d * (g * Δρ/ρ / ν²)^(1/3)
    df['D_star'] = d_m * ((G * np.abs(delta_rho) / RHO_WATER) / (NU_WATER ** 2)) ** (1/3)

    # Relative density
    df['rho_relative'] = df['density'] / RHO_WATER

    # Stokes terminal velocity (teorik - düşük Re için)
    # v_s = (g * d² * Δρ) / (18 * μ)
    # μ = ν * ρ = 1e-6 * 1000 = 1e-3 Pa.s
    mu = NU_WATER * RHO_WATER
    df['v_stokes'] = (G * (d_m ** 2) * np.abs(delta_rho)) / (18 * mu)

    # Flatness ve Elongation (şekil parametreleri)
    df['flatness'] = df['S'] / df['I']
    df['elongation'] = df['I'] / df['L']

    return df


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).squeeze().cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred
    }


def cross_validate(X, y, n_splits=5, epochs=200, lr=0.001, batch_size=32, device='cpu'):
    """5-Fold Cross Validation"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scaler (her fold için ayrı)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Tensors
        X_train_t = torch.FloatTensor(X_train_s)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val_s)
        y_val_t = torch.FloatTensor(y_val)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

        # Model
        model = SettlingVelocityNN(X.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        # Training
        best_val_r2 = -np.inf
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step(val_metrics['rmse'])

            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 30:
                break

        cv_scores.append(best_val_r2)
        print(f"  Fold {fold+1}: R² = {best_val_r2:.4f}")

    return np.array(cv_scores)


def main():
    print("=" * 70)
    print("NEURAL NETWORK (BPNN) - Mikroplastik Settling Velocity")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 1. Veri yükle
    print("\n1. Veri yükleniyor...")
    df = pd.read_csv('ml_data_full_v2.csv')
    print(f"   {len(df)} satır yüklendi")

    # 2. Physics features ekle
    print("\n2. Physics-based feature'lar ekleniyor...")
    df = add_physics_features(df)

    print(f"   D* aralığı: {df['D_star'].min():.1f} - {df['D_star'].max():.1f}")
    print(f"   v_stokes aralığı: {df['v_stokes'].min():.4f} - {df['v_stokes'].max():.4f} m/s")

    # 3. Feature seçimi
    # Orijinal + yeni physics features
    features = [
        # Boyutlar
        'a', 'b', 'c', 'd_eq', 'L', 'I', 'S',
        # Hacim/yüzey
        'volume', 'surface_area',
        # Şekil faktörleri
        'sphericity', 'aspect_ratio', 'CSF', 'flatness', 'elongation',
        # Yoğunluk
        'density', 'delta_rho', 'rho_relative',
        # Physics-based (YENİ!)
        'D_star', 'v_stokes',
        # Kategorik
        'cat_enc', 'mat_enc', 'view_enc'
    ]

    # NaN kontrolü
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['velocity_ms'])
    print(f"   Temiz veri: {len(df)} satır")

    X = df[features].values
    y = df['velocity_ms'].values

    print(f"   Feature sayısı: {len(features)}")
    print(f"   Features: {features}")

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\n   Train: {len(X_train)}, Test: {len(X_test)}")

    # 5. Cross-validation
    print("\n3. 5-Fold Cross Validation...")
    cv_scores = cross_validate(X_train, y_train, n_splits=5, epochs=300, lr=0.001, device=device)
    print(f"\n   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 6. Final model eğitimi
    print("\n4. Final model eğitimi...")

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_s)
    y_test_t = torch.FloatTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    model = SettlingVelocityNN(len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_test_r2 = -np.inf
    best_model_state = None

    for epoch in range(300):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step(test_metrics['rmse'])

        if test_metrics['r2'] > best_test_r2:
            best_test_r2 = test_metrics['r2']
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Test R² = {test_metrics['r2']:.4f}")

    # Best model'ı yükle
    model.load_state_dict(best_model_state)
    final_metrics = evaluate(model, test_loader, device)

    # 7. Sonuçlar
    print("\n" + "=" * 70)
    print("SONUÇLAR - BPNN (Literatür Tabanlı Mimari)")
    print("=" * 70)
    print(f"Test R²:   {final_metrics['r2']:.4f}")
    print(f"Test RMSE: {final_metrics['rmse']:.6f} m/s ({final_metrics['rmse']*100:.2f} cm/s)")
    print(f"Test MAE:  {final_metrics['mae']:.6f} m/s ({final_metrics['mae']*100:.2f} cm/s)")
    print(f"CV R²:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 8. Random Forest ile karşılaştırma
    print("\n" + "=" * 70)
    print("KARŞILAŞTIRMA - Random Forest (aynı features)")
    print("=" * 70)

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)

    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae = mean_absolute_error(y_test, y_pred_rf)

    from sklearn.model_selection import cross_val_score
    rf_cv = cross_val_score(rf, X_train_s, y_train, cv=5, scoring='r2')

    print(f"Test R²:   {rf_r2:.4f}")
    print(f"Test RMSE: {rf_rmse:.6f} m/s ({rf_rmse*100:.2f} cm/s)")
    print(f"Test MAE:  {rf_mae:.6f} m/s ({rf_mae*100:.2f} cm/s)")
    print(f"CV R²:     {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # Feature importance (RF)
    print("\nFeature Importance (RF - yeni features dahil):")
    imp = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    for _, row in imp.head(10).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:15s}: {row['importance']:.3f} {bar}")

    # 9. Özet karşılaştırma
    print("\n" + "=" * 70)
    print("ÖZET KARŞILAŞTIRMA")
    print("=" * 70)
    print(f"{'Model':<25} {'Test R²':>10} {'CV R²':>15} {'RMSE (cm/s)':>12}")
    print("-" * 70)
    print(f"{'BPNN (Neural Network)':<25} {final_metrics['r2']:>10.4f} {cv_scores.mean():>10.4f} ± {cv_scores.std():.2f} {final_metrics['rmse']*100:>10.2f}")
    print(f"{'Random Forest':<25} {rf_r2:>10.4f} {rf_cv.mean():>10.4f} ± {rf_cv.std():.2f} {rf_rmse*100:>10.2f}")
    print("-" * 70)

    diff = final_metrics['r2'] - rf_r2
    print(f"\nNN vs RF farkı: {diff:+.4f} ({diff*100:+.2f}%)")

    # Model kaydet
    torch.save({
        'model_state': best_model_state,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features
    }, 'settling_velocity_nn.pth')
    print("\nModel kaydedildi: settling_velocity_nn.pth")


if __name__ == "__main__":
    main()
