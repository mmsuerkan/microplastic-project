"""
Final Verification - Training Data vs Original Excel
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

# Training data yükle
train_df = pd.read_csv(r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\data\training_data_v2.csv')

print(f"\nToplam kayit: {len(train_df)}")

# Shape dağılımı
print("\n" + "=" * 70)
print("SHAPE DAGILIMI")
print("=" * 70)
for shape in sorted(train_df['shape_name'].unique()):
    count = len(train_df[train_df['shape_name'] == shape])
    print(f"  {shape:25s}: {count:4d} kayit")

# Her shape için örnek veriler
print("\n" + "=" * 70)
print("ORNEK VERILER (her shape'den 3'er)")
print("=" * 70)

for shape in sorted(train_df['shape_name'].unique()):
    print(f"\n{shape}:")
    samples = train_df[train_df['shape_name'] == shape].head(3)
    for _, row in samples.iterrows():
        print(f"  {row['category']:20s} | {row['code']:10s} | a={row['a']:6.2f}, b={row['b']:6.2f}, c={row['c']:6.2f}, d={row['density']:7.0f}, v={row['velocity_cms']:5.2f} cm/s")

# İstatistikler
print("\n" + "=" * 70)
print("ISTATISTIKLER")
print("=" * 70)

for col in ['a', 'b', 'c', 'density', 'velocity_cms']:
    print(f"\n{col}:")
    print(f"  Min: {train_df[col].min():.2f}")
    print(f"  Max: {train_df[col].max():.2f}")
    print(f"  Mean: {train_df[col].mean():.2f}")
    print(f"  Std: {train_df[col].std():.2f}")

# RESIN verileri özel kontrol
print("\n" + "=" * 70)
print("RESIN VERILERI OZEL KONTROL")
print("=" * 70)

resin_df = train_df[train_df['category'].str.contains('RESIN', na=False)]
print(f"\nToplam RESIN kayit: {len(resin_df)}")

for cat in resin_df['category'].unique():
    cat_df = resin_df[resin_df['category'] == cat]
    print(f"\n{cat}:")
    for shape in cat_df['shape_name'].unique():
        shape_df = cat_df[cat_df['shape_name'] == shape]
        print(f"  {shape}: {len(shape_df)} kayit")
        sample = shape_df.iloc[0]
        print(f"    Ornek: a={sample['a']:.2f}, b={sample['b']:.2f}, c={sample['c']:.2f}, d={sample['density']:.0f}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
