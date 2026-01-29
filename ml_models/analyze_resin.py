"""RESIN parçacıklarının detaylı analizi"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Hem ortalama hem de tüm ölçümleri yükle
df_avg = pd.read_csv('data/training_data_particle_avg.csv')
df_all = pd.read_csv('data/training_data_v2_features.csv')

print("="*70)
print("RESIN PARÇACİKLARİ ANALİZİ")
print("="*70)

# RESIN parçacıkları
resin = df_avg[df_avg['category'].str.contains('RESIN')]
others = df_avg[~df_avg['category'].str.contains('RESIN')]

print(f"\nRESIN parçacık sayısı: {len(resin)}")
print(f"Diğer parçacık sayısı: {len(others)}")

# ================================================================
# 1. RESIN vs DİĞERLERİ KARŞILAŞTIRMASI
# ================================================================
print("\n" + "="*70)
print("1. RESIN vs DİĞER MATERYALLER")
print("="*70)

print("\n--- Velocity Karşılaştırması ---")
print(f"RESIN ortalama velocity: {resin['velocity_mean'].mean():.2f} ± {resin['velocity_mean'].std():.2f} cm/s")
print(f"Diğerleri ortalama:      {others['velocity_mean'].mean():.2f} ± {others['velocity_mean'].std():.2f} cm/s")
print(f"Fark:                    {resin['velocity_mean'].mean() - others['velocity_mean'].mean():.2f} cm/s ({resin['velocity_mean'].mean() / others['velocity_mean'].mean():.1f}x)")

print("\n--- Density Karşılaştırması ---")
print(f"RESIN ortalama density: {resin['density'].mean():.0f} kg/m³")
print(f"Diğerleri ortalama:     {others['density'].mean():.0f} kg/m³")

print("\n--- Boyut Karşılaştırması ---")
print(f"RESIN ortalama 'a': {resin['a'].mean():.2f} mm")
print(f"Diğerleri ortalama: {others['a'].mean():.2f} mm")

# ================================================================
# 2. RESIN DETAYLARI (Kategori bazlı)
# ================================================================
print("\n" + "="*70)
print("2. RESIN KATEGORİLERİ DETAYI")
print("="*70)

resin_detail = resin.groupby('category').agg({
    'velocity_mean': ['mean', 'std', 'min', 'max', 'count'],
    'density': 'mean',
    'a': 'mean'
}).round(2)
resin_detail.columns = ['Vel Mean', 'Vel Std', 'Vel Min', 'Vel Max', 'Count', 'Density', 'Size a']
print(resin_detail.to_string())

# ================================================================
# 3. RESIN ŞEKİL BAZLI ANALİZ
# ================================================================
print("\n" + "="*70)
print("3. RESIN ŞEKİL BAZLI ANALİZ")
print("="*70)

resin_shape = resin.groupby('shape_name').agg({
    'velocity_mean': ['mean', 'std', 'count'],
    'density': 'mean'
}).round(2)
resin_shape.columns = ['Vel Mean', 'Vel Std', 'Count', 'Density']
resin_shape = resin_shape.sort_values('Vel Mean', ascending=False)
print(resin_shape.to_string())

# ================================================================
# 4. AYNI ŞEKİL, FARKLI MATERYAL KARŞILAŞTIRMASI
# ================================================================
print("\n" + "="*70)
print("4. AYNI ŞEKİL - FARKLI MATERYAL (Sphere)")
print("="*70)

spheres = df_avg[df_avg['shape_name'] == 'Sphere']
print("\n--- Tüm Sphere'ler ---")
sphere_detail = spheres[['category', 'code', 'a', 'density', 'velocity_mean']].sort_values('velocity_mean', ascending=False)
print(sphere_detail.to_string(index=False))

print("\n--- Materyal bazlı Sphere ortalaması ---")
sphere_by_mat = spheres.groupby(spheres['category'].str.split().str[0]).agg({
    'velocity_mean': ['mean', 'count'],
    'density': 'mean',
    'a': 'mean'
}).round(2)
sphere_by_mat.columns = ['Vel Mean', 'Count', 'Density', 'Size']
print(sphere_by_mat.to_string())

# ================================================================
# 5. RESIN ÖLÇÜM VARYANSI (Tekrarlı ölçümler)
# ================================================================
print("\n" + "="*70)
print("5. RESIN ÖLÇÜM VARYANSI (Aynı parçacığın tekrarlı ölçümleri)")
print("="*70)

# Tüm ölçümlerden RESIN'leri al
resin_all = df_all[df_all['category'].str.contains('RESIN')]
resin_variance = resin_all.groupby(['category', 'code']).agg({
    'velocity_cms': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
resin_variance.columns = ['Mean', 'Std', 'Min', 'Max', 'Count']
resin_variance = resin_variance.sort_values('Std', ascending=False)

print("\n--- En Yüksek Varyanslı RESIN Parçacıkları ---")
print(resin_variance.head(15).to_string())

# Yüksek varyans analizi
high_var = resin_variance[resin_variance['Std'] > 1.5]
print(f"\nStd > 1.5 cm/s olan: {len(high_var)} parçacık")

# ================================================================
# 6. FİZİKSEL ANALİZ
# ================================================================
print("\n" + "="*70)
print("6. FİZİKSEL ANALİZ - Neden RESIN hızlı?")
print("="*70)

# Stokes Law: v = (2/9) * (ρp - ρf) * g * r² / μ
# ρp: parçacık density, ρf: sıvı density (~1000), g: 9.81, r: yarıçap, μ: viskozite

print("""
Stokes Law'a göre settling velocity:
  v ∝ (ρ_particle - ρ_fluid) × r²

Faktörler:
  1. Density farkı (ρp - ρf): Yüksek density → Hızlı batma
  2. Boyut (r²): Büyük parçacık → Hızlı batma
  3. Şekil: Sphere en aerodinamik → En hızlı
""")

# RESIN vs diğerleri fiziksel karşılaştırma
print("\n--- Fiziksel Karşılaştırma ---")
resin_spheres = df_avg[(df_avg['category'].str.contains('RESIN')) & (df_avg['shape_name'] == 'Sphere')]
other_spheres = df_avg[(~df_avg['category'].str.contains('RESIN')) & (df_avg['shape_name'] == 'Sphere')]

if len(resin_spheres) > 0 and len(other_spheres) > 0:
    print(f"\nSphere karşılaştırması:")
    print(f"                    RESIN Sphere    Diğer Sphere")
    print(f"  Density:          {resin_spheres['density'].mean():.0f} kg/m³       {other_spheres['density'].mean():.0f} kg/m³")
    print(f"  Boyut (a):        {resin_spheres['a'].mean():.1f} mm          {other_spheres['a'].mean():.1f} mm")
    print(f"  Velocity:         {resin_spheres['velocity_mean'].mean():.1f} cm/s        {other_spheres['velocity_mean'].mean():.1f} cm/s")

# Density vs Velocity korelasyonu
print("\n--- Density-Velocity İlişkisi (Sphere'ler) ---")
spheres_corr = spheres[['density', 'a', 'velocity_mean']].corr()['velocity_mean']
print(f"  Density-Velocity korelasyonu: {spheres_corr['density']:.3f}")
print(f"  Boyut-Velocity korelasyonu:   {spheres_corr['a']:.3f}")

# ================================================================
# 7. SONUÇ VE ÖNERİLER
# ================================================================
print("\n" + "="*70)
print("7. SONUÇ VE ÖNERİLER")
print("="*70)

resin_sphere_vel = resin_spheres['velocity_mean'].mean() if len(resin_spheres) > 0 else 0
other_sphere_vel = other_spheres['velocity_mean'].mean() if len(other_spheres) > 0 else 0

print(f"""
BULGULAR:
---------
1. RESIN Sphere velocity ({resin_sphere_vel:.1f} cm/s) diğerlerinden
   ({other_sphere_vel:.1f} cm/s) çok yüksek

2. RESIN parçacıkları:
   - Daha BÜYÜK (a={resin['a'].mean():.1f} mm vs {others['a'].mean():.1f} mm)
   - Density benzer ({resin['density'].mean():.0f} vs {others['density'].mean():.0f} kg/m³)

3. Büyük boyut = Yüksek velocity (Stokes law)
   Bu FİZİKSEL olarak DOĞRU!

ÖNERİLER:
---------
A. ÖLÇÜM HATASI DEĞİL - Fiziksel olarak açıklanabilir
   - Büyük RESIN parçacıkları gerçekten hızlı batıyor

B. MODEL İYİLEŞTİRME:
   - Boyut etkisini daha iyi yakalamak için:
     * Reynolds number ekle
     * Boyut²/Density oranı ekle

C. VERİ DENGESİZLİĞİ:
   - RESIN büyük boyutlu, diğerleri küçük
   - Model küçük boyutları öğrenmiş, büyükleri tahmin edemiyor
   - Çözüm: Boyut bazlı stratified sampling veya ağırlıklı loss
""")
