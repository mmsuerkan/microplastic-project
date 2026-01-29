"""Yeni verileri eski feature dosyasına ekle (feature formüllerini koru)"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Eski dosya (doğru feature'lar, 0.84 R²)
old_df = pd.read_csv('data/training_data_v2_features.csv')
print(f"Eski dosya: {len(old_df)} satır")

# Yeni dosya (yeni veriler dahil)
new_df = pd.read_csv('data/training_data_v2.csv')
print(f"Yeni dosya: {len(new_df)} satır")

# Unique key oluştur: category + code + date + view + repeat
old_df['key'] = old_df['category'] + '_' + old_df['code'] + '_' + old_df['date'] + '_' + old_df['view'] + '_' + old_df['repeat']
new_df['key'] = new_df['category'] + '_' + new_df['code'] + '_' + new_df['date'] + '_' + new_df['view'] + '_' + new_df['repeat']

# Eski dosyada olmayan yeni verileri bul
old_keys = set(old_df['key'])
new_entries = new_df[~new_df['key'].isin(old_keys)].copy()
print(f"\nYeni veri sayısı: {len(new_entries)}")

if len(new_entries) > 0:
    print("\nYeni veriler:")
    print(new_entries[['category', 'code', 'date', 'view', 'repeat', 'velocity_cms']].to_string())

    # Eski dosyadaki feature formüllerini kullan
    # Eski dosyadan örnek al ve formülleri kontrol et
    print("\n" + "="*60)
    print("ESKİ DOSYA FEATURE ÖRNEKLERİ (doğru formüller)")
    print("="*60)

    # Cylinder örneği
    cyl = old_df[old_df['shape_enc'] == 0].head(1)
    if len(cyl) > 0:
        row = cyl.iloc[0]
        print(f"\nCylinder: a={row['a']}, b={row['b']}, c={row['c']}")
        print(f"  volume={row['volume']:.4f}")
        print(f"  Formül: π*(b/2)²*a = {np.pi * (row['b']/2)**2 * row['a']:.4f}" if row['b'] > 0 else "b=0")

    # Sphere örneği
    sph = old_df[old_df['shape_enc'] == 5].head(1)
    if len(sph) > 0:
        row = sph.iloc[0]
        print(f"\nSphere: a={row['a']}")
        print(f"  volume={row['volume']:.4f}")
        print(f"  Formül: (4/3)*π*(a/2)³ = {(4/3) * np.pi * (row['a']/2)**3:.4f}")

    # Yeni veriler için feature hesapla (eski formüllerle)
    def calc_volume_old(row):
        shape = row['shape_enc']
        a, b, c = row['a'], row['b'], row['c']
        if shape == 0:  # Cylinder: a=cap, b=yukseklik
            return np.pi * (a/2)**2 * b if b > 0 else np.pi * (a/2)**2 * a
        elif shape == 1:  # Half Cylinder
            return 0.5 * np.pi * (a/2)**2 * b if b > 0 else 0.5 * np.pi * (a/2)**2 * a
        elif shape == 2:  # Cube
            return a * b * c if b > 0 and c > 0 else a**3
        elif shape == 3:  # Wedge
            return 0.5 * a * b * c if b > 0 and c > 0 else 0.5 * a**3
        elif shape == 4:  # Box
            return a * b * c if b > 0 and c > 0 else a**3
        elif shape == 5:  # Sphere
            return (4/3) * np.pi * (a/2)**3
        elif shape == 6:  # Elliptic Cylinder
            return np.pi * (a/2) * (b/2) * c if b > 0 and c > 0 else np.pi * (a/2)**2 * a
        return a * b * c if b > 0 and c > 0 else a**3

    def calc_surface_area_old(row):
        shape = row['shape_enc']
        a, b, c = row['a'], row['b'], row['c']
        if shape == 5:  # Sphere
            return 4 * np.pi * (a/2)**2
        elif shape == 0:  # Cylinder: a=cap, b=yukseklik
            r = a/2  # a = cap
            h = b if b > 0 else a  # b = yukseklik
            return 2 * np.pi * r * (r + h)
        return 2 * (a*b + b*c + a*c) if b > 0 and c > 0 else 6 * a**2

    # Feature'ları hesapla
    new_entries['volume'] = new_entries.apply(calc_volume_old, axis=1)
    new_entries['surface_area'] = new_entries.apply(calc_surface_area_old, axis=1)
    new_entries['aspect_ratio'] = new_entries['a'] / np.where(new_entries['b'] == 0, new_entries['a'], new_entries['b'])
    new_entries['vol_surf_ratio'] = new_entries['volume'] / np.where(new_entries['surface_area'] == 0, 1, new_entries['surface_area'])

    # Key sütununu kaldır
    new_entries = new_entries.drop('key', axis=1)
    old_df = old_df.drop('key', axis=1)

    # Birleştir
    merged_df = pd.concat([old_df, new_entries], ignore_index=True)
    print(f"\n" + "="*60)
    print(f"BİRLEŞTİRME SONUCU")
    print("="*60)
    print(f"Eski: {len(old_df)} + Yeni: {len(new_entries)} = Toplam: {len(merged_df)}")

    # Kaydet
    merged_df.to_csv('data/training_data_v2_features.csv', index=False)
    print(f"\nKaydedildi: data/training_data_v2_features.csv")

    # Unique parçacık kontrolü
    unique_particles = merged_df.groupby(['category', 'code']).size()
    print(f"Unique parçacık sayısı: {len(unique_particles)}")
else:
    print("Yeni veri yok!")
    old_df = old_df.drop('key', axis=1)
