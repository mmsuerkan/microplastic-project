"""
Basitleştirilmiş Model - 5 Girdi
Girdiler: a, b, c, density, shape
Şekiller: Cylinder, Half Cylinder, Cube, Wedge, Box Prism, Sphere, Elliptic Cylinder
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

# Şekil kodları
SHAPE_CODES = {
    0: 'Cylinder',
    1: 'Half Cylinder',
    2: 'Cube',
    3: 'Wedge Shape Prism',
    4: 'Box Shape Prism',
    5: 'Sphere',
    6: 'Elliptic Cylinder'
}

# Kategori -> Şekil eşleştirmesi
CATEGORY_TO_SHAPE = {
    'ABS C': 0,           # Cylinder
    'PLA C': 0,           # Cylinder
    'C': 0,               # Cylinder
    'ABS HC': 1,          # Half Cylinder
    'PLA HC': 1,          # Half Cylinder
    'HC': 1,              # Half Cylinder
    'PLA CUBE': 2,        # Cube
    'WSP': 3,             # Wedge Shape Prism (PMMA Wedge-Shaped)
    'BSP': 4,             # Box Shape Prism
    'PS': 6,              # Elliptic Cylinder (PS EC)
    'PA 6': 4,            # Box Shape Prism (P6 BSP benzeri)
    # RESIN'ler ayrı işlenecek
}


class SimpleNN(nn.Module):
    """Basit Neural Network - 5 girdi için"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        # 5 girdi: a, b, c, density, shape_enc
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 128),
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


def map_category_to_shape(row):
    """Kategoriyi şekil koduna dönüştür"""
    category = row['category']

    # RESIN kontrolü
    if 'RESIN' in category:
        # RESIN (A=6 R=3) veya RESIN (A=9 R=4.5) formatında
        if 'R=' in category:
            # r= olanlar küre
            return 5  # Sphere
        elif 'A=' in category:
            # a= olanlar küp
            return 2  # Cube

    # Diğer kategoriler
    return CATEGORY_TO_SHAPE.get(category, 4)  # Default: Box Shape Prism


def transform_dimensions(row):
    """Şekle göre a, b, c dönüşümü"""
    shape = row['shape_enc']
    a, b, c = row['a'], row['b'], row['c']

    if shape == 5:  # Sphere
        # a = çap (2r), b = 0, c = 0
        diameter = max(a, b, c)  # En büyük boyut çap
        return diameter, 0.0, 0.0

    elif shape in [0, 6]:  # Cylinder, Elliptic Cylinder
        # a = çap, b = yükseklik, c = 0
        # Mevcut veride hangisi çap hangisi yükseklik belirsiz
        # En kısa 2 boyutun ortalaması çap, en uzun yükseklik olabilir
        dims = sorted([a, b, c])
        diameter = (dims[0] + dims[1]) / 2 if dims[0] > 0 else dims[1]
        height = dims[2]
        return diameter, height, 0.0

    elif shape == 1:  # Half Cylinder
        # 3 boyut da geçerli
        return a, b, c

    else:  # Cube, Wedge, Box Prism
        # 3 boyut da geçerli
        return a, b, c


def prepare_data(df):
    """Veriyi yeni formata dönüştür"""
    df = df.copy()

    # Şekil kodunu ekle
    df['shape_enc'] = df.apply(map_category_to_shape, axis=1)

    # Boyutları dönüştür
    transformed = df.apply(transform_dimensions, axis=1)
    df['a_new'] = [t[0] for t in transformed]
    df['b_new'] = [t[1] for t in transformed]
    df['c_new'] = [t[2] for t in transformed]

    return df


def cross_validate_ensemble(X, y, n_splits=5):
    """5-Fold CV for Ensemble"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rf_scores = []
    nn_scores = []
    ensemble_scores = []

    device = torch.device('cpu')

    print("\n5-Fold Cross Validation:")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_val_s)
        rf_r2 = r2_score(y_val, rf_pred)
        rf_scores.append(rf_r2)

        # Neural Network
        X_train_t = torch.FloatTensor(X_train_s)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val_s)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_val_r2 = -np.inf
        patience = 0

        for epoch in range(300):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch).squeeze(), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                nn_pred_val = model(X_val_t).squeeze().numpy()
            val_r2 = r2_score(y_val, nn_pred_val)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience = 0
            else:
                patience += 1
                if patience >= 30:
                    break

        nn_scores.append(best_val_r2)

        # Ensemble
        model.eval()
        with torch.no_grad():
            nn_pred = model(X_val_t).squeeze().numpy()
        ensemble_pred = (rf_pred + nn_pred) / 2
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        ensemble_scores.append(ensemble_r2)

        print(f"  Fold {fold+1}: RF={rf_r2:.4f}, NN={best_val_r2:.4f}, Ensemble={ensemble_r2:.4f}")

    print("-" * 60)
    print(f"  Mean:   RF={np.mean(rf_scores):.4f}, NN={np.mean(nn_scores):.4f}, Ensemble={np.mean(ensemble_scores):.4f}")

    return {
        'rf': np.array(rf_scores),
        'nn': np.array(nn_scores),
        'ensemble': np.array(ensemble_scores)
    }


def train_final_model(X, y):
    """Final model eğitimi"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    print("\n1. Random Forest eğitiliyor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_test_pred = rf.predict(X_test_s)
    print(f"   Test R²: {r2_score(y_test, rf_test_pred):.4f}")

    # Neural Network
    print("\n2. Neural Network eğitiliyor...")
    device = torch.device('cpu')

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_s)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_state = None
    best_test_r2 = -np.inf

    for epoch in range(300):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            nn_test_pred = model(X_test_t).squeeze().numpy()
        test_r2 = r2_score(y_test, nn_test_pred)
        scheduler.step(-test_r2)

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    print(f"   Test R²: {best_test_r2:.4f}")

    # Ensemble
    print("\n3. Ensemble (RF + NN):")
    model.eval()
    with torch.no_grad():
        nn_test_pred = model(X_test_t).squeeze().numpy()

    ensemble_pred = (rf_test_pred + nn_test_pred) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

    print(f"   Test R²:  {ensemble_r2:.4f}")
    print(f"   Test RMSE: {ensemble_rmse*100:.2f} cm/s")

    return rf, model, scaler


def main():
    print("=" * 70)
    print("BASİTLEŞTİRİLMİŞ MODEL - 5 Girdi")
    print("Girdiler: a, b, c, density, shape")
    print("=" * 70)

    # Veri yükle - YENİ training_data_v2.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'training_data_v2.csv')

    print(f"\nVeri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Toplam: {len(df)} satır")

    # Şekil dağılımı
    print("\nŞekil Dağılımı:")
    for code, name in SHAPE_CODES.items():
        count = len(df[df['shape_enc'] == code])
        if count > 0:
            print(f"  {code}: {name} - {count} adet")

    # Feature'lar - doğrudan a, b, c kullan
    features = ['a', 'b', 'c', 'density', 'shape_enc']

    X = df[features].values
    y = df['velocity_ms'].values

    print(f"\nFeature sayısı: {len(features)}")
    print(f"Features: {features}")

    # Cross Validation
    cv_scores = cross_validate_ensemble(X, y, n_splits=5)

    print("\n" + "=" * 70)
    print("CROSS VALIDATION SONUÇLARI")
    print("=" * 70)
    print(f"Random Forest:  {cv_scores['rf'].mean():.4f} ± {cv_scores['rf'].std():.4f}")
    print(f"Neural Network: {cv_scores['nn'].mean():.4f} ± {cv_scores['nn'].std():.4f}")
    print(f"ENSEMBLE:       {cv_scores['ensemble'].mean():.4f} ± {cv_scores['ensemble'].std():.4f}")

    # Final model
    print("\n" + "=" * 70)
    print("FİNAL MODEL EĞİTİMİ")
    print("=" * 70)
    rf, nn_model, scaler = train_final_model(X, y)

    # Kaydet
    print("\n" + "=" * 70)
    print("MODEL KAYDEDİLİYOR")
    print("=" * 70)

    # RF kaydet
    rf_path = os.path.join(script_dir, 'simple_rf.joblib')
    joblib.dump(rf, rf_path)
    print(f"RF kaydedildi: {rf_path}")

    # NN kaydet
    nn_path = os.path.join(script_dir, 'simple_nn.pth')
    torch.save({
        'model_state': nn_model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features,
        'shape_codes': SHAPE_CODES
    }, nn_path)
    print(f"NN kaydedildi: {nn_path}")

    # Meta kaydet
    meta_path = os.path.join(script_dir, 'simple_meta.joblib')
    joblib.dump({
        'features': features,
        'shape_codes': SHAPE_CODES,
        'cv_scores': cv_scores
    }, meta_path)
    print(f"Meta kaydedildi: {meta_path}")

    print("\n" + "=" * 70)
    print("KULLANIM")
    print("=" * 70)
    print("""
Girdiler:
  - a: 1. boyut (mm) - Sphere için çap, Cylinder için çap
  - b: 2. boyut (mm) - Sphere için 0, Cylinder için yükseklik
  - c: 3. boyut (mm) - Sphere/Cylinder için 0
  - density: Yoğunluk (kg/m³)
  - shape: Şekil kodu (0-6)

Şekil Kodları:
  0: Cylinder
  1: Half Cylinder
  2: Cube
  3: Wedge Shape Prism
  4: Box Shape Prism
  5: Sphere
  6: Elliptic Cylinder
""")


if __name__ == "__main__":
    main()
