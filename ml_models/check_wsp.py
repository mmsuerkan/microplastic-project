import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results/experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print('=== WSP EXPERIMENTS ===')
codes = set()
for exp in data['experiments']:
    if exp['category'] == 'WSP' and exp['status'] == 'success':
        codes.add(exp['code'])
for c in sorted(codes):
    print(f'  {c}')
