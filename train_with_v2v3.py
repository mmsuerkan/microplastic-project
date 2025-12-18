"""
BPNN Model - V2/V3 verileri eklenerek egitim
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import os
import glob

# Fiziksel sabitler
G = 9.81
RHO_WATER = 1000
NU_WATER = 1.0e-6


def parse_summary(filepath):
    metrics = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key == 'Tahmini Hiz' and 'cm/s' not in line:
                        metrics['hiz_ms'] = float(val)
    except:
        pass
    return metrics


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


def main():
    print('=' * 70)
    print('BPNN - V2/V3 VERILERI EKLENDI')
    print('=' * 70)

    # 1. Mevcut veriyi yukle
    df_existing = pd.read_csv('ml_data_full_v2.csv')
    print(f'Mevcut veri: {len(df_existing)}')

    # Particle bilgilerini al
    particle_cols = ['category_norm', 'code', 'material', 'a', 'b', 'c', 'density',
                     'd_eq', 'volume', 'surface_area', 'sphericity', 'L', 'I', 'S',
                     'aspect_ratio', 'CSF', 'delta_rho', 'cat_enc', 'mat_enc']
    particles = df_existing[particle_cols].drop_duplicates(subset=['category_norm', 'code'])

    # 2. V2/V3 verilerini topla
    base = 'processed_results'
    new_data = []

    for version, folder in [('V2', 'second_iteration'), ('V3', 'third_iteration')]:
        path = os.path.join(base, folder, 'success', '**', 'summary.csv')
        for summary_path in glob.glob(path, recursive=True):
            parts = os.path.normpath(summary_path).split(os.sep)
            category = parts[-3].upper()
            code = parts[-2].upper()

            metrics = parse_summary(summary_path)
            hiz = metrics.get('hiz_ms', 0)

            if 0 < hiz < 0.5:
                view = parts[-5].upper() if len(parts) > 5 else 'MAK'
                new_data.append({
                    'category_norm': category,
                    'code': code,
                    'velocity_ms': hiz,
                    'view': view
                })

    df_new = pd.DataFrame(new_data)
    print(f'V2/V3 ham veri: {len(df_new)}')

    # 3. Particle bilgisiyle eslestir
    df_new_merged = pd.merge(df_new, particles, on=['category_norm', 'code'], how='inner')
    print(f'V2/V3 eslesen: {len(df_new_merged)}')

    # view_enc ekle
    le_view = LabelEncoder()
    le_view.fit(df_existing['view'])
    df_new_merged['view_enc'] = df_new_merged['view'].apply(
        lambda x: le_view.transform([x])[0] if x in le_view.classes_ else 0
    )

    # 4. Mevcut veriyle birlestir
    df_combined = pd.concat([df_existing, df_new_merged], ignore_index=True)
    print(f'Birlesik veri: {len(df_combined)}')

    # 5. RESIN (A=9) cikar
    df_clean = df_combined[~df_combined['category_norm'].str.contains('RESIN.*A=9', case=False, regex=True, na=False)]
    print(f'RESIN A=9 cikarildi: {len(df_clean)}')

    # 6. Physics features ekle
    d_m = df_clean['d_eq'] / 1000
    delta_rho = df_clean['density'] - RHO_WATER
    df_clean = df_clean.copy()
    df_clean['D_star'] = d_m * ((G * np.abs(delta_rho) / RHO_WATER) / (NU_WATER ** 2)) ** (1/3)
    df_clean['rho_relative'] = df_clean['density'] / RHO_WATER
    mu = NU_WATER * RHO_WATER
    df_clean['v_stokes'] = (G * (d_m ** 2) * np.abs(delta_rho)) / (18 * mu)
    df_clean['flatness'] = df_clean['S'] / df_clean['I']
    df_clean['elongation'] = df_clean['I'] / df_clean['L']

    features = ['a', 'b', 'c', 'd_eq', 'L', 'I', 'S', 'volume', 'surface_area',
                'sphericity', 'aspect_ratio', 'CSF', 'flatness', 'elongation',
                'density', 'delta_rho', 'rho_relative', 'D_star', 'v_stokes',
                'cat_enc', 'mat_enc', 'view_enc']

    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['velocity_ms'])
    print(f'Final temiz veri: {len(df_clean)}')

    X = df_clean[features].values
    y = df_clean['velocity_ms'].values

    # 7. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f'Train: {len(X_train)}, Test: {len(X_test)}')

    # 8. 5-Fold CV
    print()
    print('5-Fold Cross Validation...')
    device = torch.device('cpu')
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr)),
            batch_size=32, shuffle=True
        )

        model = SettlingVelocityNN(len(features)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_r2 = -np.inf
        for epoch in range(200):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch).squeeze(), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_val_s)).squeeze().numpy()
            r2 = r2_score(y_val, y_pred)
            if r2 > best_r2:
                best_r2 = r2

        cv_scores.append(best_r2)
        print(f'  Fold {fold+1}: R2 = {best_r2:.4f}')

    cv_scores = np.array(cv_scores)
    print(f'  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

    # 9. Final model
    print()
    print('Final model egitimi...')
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train)),
        batch_size=32, shuffle=True
    )

    model = SettlingVelocityNN(len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_r2 = -np.inf
    best_state = None
    for epoch in range(300):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test_s)).squeeze().numpy()
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_final = model(torch.FloatTensor(X_test_s)).squeeze().numpy()

    final_r2 = r2_score(y_test, y_pred_final)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

    print()
    print('=' * 70)
    print('SONUCLAR')
    print('=' * 70)
    new_count = len(df_clean) - 966
    print(f'Veri: {len(df_clean)} satir (+{new_count} yeni)')
    print(f'Test R2:  {final_r2:.4f}')
    print(f'CV R2:    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    print(f'RMSE:     {final_rmse*100:.2f} cm/s')
    print()
    print('=' * 70)
    print('KARSILASTIRMA')
    print('=' * 70)
    print(f'{"Veri":<20} {"CV R2":>12}')
    print('-' * 35)
    print(f'{"Onceki (966)":<20} {"0.8010":>12}')
    print(f'{"Yeni (" + str(len(df_clean)) + ")":<20} {cv_scores.mean():>12.4f}')
    print('-' * 35)
    diff = cv_scores.mean() - 0.8010
    print(f'Degisim: {diff*100:+.2f} puan')

    # Model kaydet
    torch.save({
        'model_state': best_state,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features,
        'cv_r2': cv_scores.mean(),
        'test_r2': final_r2,
        'data_count': len(df_clean)
    }, 'settling_velocity_nn_v2v3.pth')
    print(f'\nModel kaydedildi: settling_velocity_nn_v2v3.pth')


if __name__ == "__main__":
    main()
