import os
import glob
RESULTS_DIR = 'c:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results'

speed_cache = {}
for f in glob.glob(os.path.join(RESULTS_DIR, '**', 'summary.csv'), recursive=True):
    rel = os.path.relpath(f, RESULTS_DIR).replace('\\', '/').split('/')
    if rel[0] == 'success' and len(rel) >= 7:
        date, view, repeat, cat, code = rel[1], rel[2], rel[3], rel[4], rel[5]
        key = f'{view}/{date}/{repeat}/{cat}/{code}'
        speed_cache[key] = 1
    elif rel[0] == 'fail' and len(rel) >= 8:
        date, view, repeat, cat, code = rel[2], rel[3], rel[4], rel[5], rel[6]
        key = f'{view}/{date}/{repeat}/{cat}/{code}'
        speed_cache[key] = 1

print(f'Toplam {len(speed_cache)} key')
print('Cache key ornekleri:')
for k in list(speed_cache.keys())[:15]:
    print(f'  {k}')
