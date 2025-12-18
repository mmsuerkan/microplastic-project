import os
import glob

RESULTS_DIR = 'c:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results'
pattern = os.path.join(RESULTS_DIR, '**', 'summary.csv')
files = glob.glob(pattern, recursive=True)

print('Ilk 5 dosya:')
for f in files[:5]:
    rel = os.path.relpath(f, RESULTS_DIR)
    parts = rel.replace('\\', '/').split('/')
    print(f'  Path: {rel}')
    print(f'  Parts: {parts}')
    print(f'  Len: {len(parts)}')

    # Icerik
    with open(f, 'r') as file:
        for line in file:
            if 'Hiz' in line:
                print(f'  Hiz satiri: {line.strip()}')
                break
    print()
