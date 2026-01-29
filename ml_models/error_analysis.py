"""Andrew Ng Style Error Analysis"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# Veri yükle
df = pd.read_csv('data/training_data_particle_avg.csv')
print(f"Veri: {len(df)} parçacık\n")

# Features
feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values

# ================================================================
# 1. ERROR ANALYSIS: Hangi örneklerde hata yapıyoruz?
# ================================================================
print("="*70)
print("1. ERROR ANALYSIS - Modelin En Çok Hata Yaptığı Örnekler")
print("="*70)

# Cross-validation prediction (her örnek için out-of-fold tahmin)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
y_pred_cv = cross_val_predict(rf, X, y, cv=5)

# Hata hesapla
df['predicted'] = y_pred_cv
df['error'] = df['velocity_mean'] - df['predicted']
df['abs_error'] = np.abs(df['error'])
df['pct_error'] = np.abs(df['error'] / df['velocity_mean']) * 100

# En büyük hatalar
print("\n--- En Büyük Hatalar (Top 10) ---")
top_errors = df.nlargest(10, 'abs_error')[['category', 'code', 'shape_name',
                                            'velocity_mean', 'predicted', 'error', 'pct_error']]
print(top_errors.to_string(index=False))

# Şekil bazlı hata analizi
print("\n--- Şekil Bazlı Ortalama Hata ---")
shape_errors = df.groupby('shape_name').agg({
    'abs_error': ['mean', 'std', 'max'],
    'pct_error': 'mean',
    'velocity_mean': 'count'
}).round(2)
shape_errors.columns = ['MAE', 'Std', 'Max Error', 'MAPE%', 'Count']
shape_errors = shape_errors.sort_values('MAE', ascending=False)
print(shape_errors.to_string())

# Kategori bazlı hata analizi
print("\n--- Materyal Bazlı Ortalama Hata ---")
# Basit kategori (ilk kelime)
df['material'] = df['category'].apply(lambda x: x.split()[0] if ' ' in x else x.split('(')[0])
material_errors = df.groupby('material').agg({
    'abs_error': ['mean', 'max'],
    'pct_error': 'mean',
    'velocity_mean': 'count'
}).round(2)
material_errors.columns = ['MAE', 'Max Error', 'MAPE%', 'Count']
material_errors = material_errors.sort_values('MAE', ascending=False)
print(material_errors.to_string())

# ================================================================
# 2. BIAS-VARIANCE ANALYSIS: High bias mı high variance mı?
# ================================================================
print("\n" + "="*70)
print("2. BIAS-VARIANCE ANALYSIS")
print("="*70)

# Training vs CV score
rf.fit(X, y)
train_score = rf.score(X, y)
cv_scores = []
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

print(f"\nTraining R²: {train_score:.4f}")
print(f"CV R²:       {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"Gap:         {train_score - np.mean(cv_scores):.4f}")

if train_score - np.mean(cv_scores) > 0.1:
    print("\n→ HIGH VARIANCE (Overfitting) - Daha fazla veri veya regularization gerekli")
elif np.mean(cv_scores) < 0.7:
    print("\n→ HIGH BIAS (Underfitting) - Daha karmaşık model veya daha iyi feature gerekli")
else:
    print("\n→ Makul denge, veri kalitesine odaklanılabilir")

# ================================================================
# 3. LEARNING CURVE: Daha fazla veri yardımcı olur mu?
# ================================================================
print("\n" + "="*70)
print("3. LEARNING CURVE ANALYSIS")
print("="*70)

train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
train_sizes_abs, train_scores, val_scores = learning_curve(
    rf, X, y, train_sizes=train_sizes, cv=5, scoring='r2', n_jobs=-1
)

print("\n--- Veri Miktarına Göre Performans ---")
print(f"{'Veri %':<10} {'Train Samples':<15} {'Train R²':<15} {'Val R²':<15}")
for size, n_samples, train_s, val_s in zip(train_sizes, train_sizes_abs, train_scores, val_scores):
    print(f"{size*100:.0f}%{'':<8} {n_samples:<15} {np.mean(train_s):.3f} ± {np.std(train_s):.3f}{'':5} {np.mean(val_s):.3f} ± {np.std(val_s):.3f}")

# Eğilim analizi
val_trend = np.mean(val_scores[-1]) - np.mean(val_scores[0])
if val_trend > 0.05:
    print("\n→ Validation skoru veri arttıkça ARTIYOR - Daha fazla veri faydalı olabilir!")
else:
    print("\n→ Validation skoru doyuma ulaşmış - Veri kalitesi veya feature'lara odaklan")

# ================================================================
# 4. FEATURE IMPORTANCE & CORRELATION
# ================================================================
print("\n" + "="*70)
print("4. FEATURE ANALYSIS")
print("="*70)

print("\n--- Feature Importance (RF) ---")
importances = rf.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"  {feat:<15} {imp:.3f} {bar}")

# Düşük önemli feature'lar
low_importance = [f for f, i in zip(feature_cols, importances) if i < 0.05]
if low_importance:
    print(f"\n→ Düşük önemli feature'lar (<5%): {low_importance}")
    print("  Bu feature'lar çıkarılabilir veya yenileriyle değiştirilebilir")

# ================================================================
# 5. ÖNERİLER
# ================================================================
print("\n" + "="*70)
print("5. ANDREW NG YAKLAŞIMI - ÖNERİLER")
print("="*70)

print("""
A. DATA QUALITY (Öncelik 1):
   - Yüksek hatalı örnekleri manuel incele (yukarıdaki Top 10)
   - Ölçüm hatası mı, gerçek anomali mi?
   - Sphere'ler çok yüksek hata → fiziksel açıklama var mı?

B. FEATURE ENGINEERING (Öncelik 2):
   - Reynolds number ekle: Re = (density * velocity * diameter) / viscosity
   - Sphericity (küresellik) ekle
   - Boyutsuz oranlar (a/b, b/c, etc.)

C. DATA AUGMENTATION (Öncelik 3):
   - Fiziksel simülasyonlardan sentetik veri
   - Benzer çalışmalardan veri toplama

D. MODEL (Son öncelik):
   - Model zaten iyi (R²=0.83)
   - Veri kalitesi > Model karmaşıklığı
""")

# Kaydet
df[['category', 'code', 'shape_name', 'velocity_mean', 'predicted', 'error', 'abs_error', 'pct_error']].to_csv(
    'data/error_analysis.csv', index=False
)
print("\nHata analizi kaydedildi: data/error_analysis.csv")
