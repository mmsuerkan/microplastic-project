"""
MAK + ANG pair finder for 3D reconstruction.

Finds experiments where the same particle (same code, repeat, date)
was recorded from both MAK and ANG views successfully.
"""
import sys
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


def find_paired_experiments(json_path):
    """
    Find experiments where the same particle was recorded from both MAK and ANG views.

    Parameters
    ----------
    json_path : str
        Path to experiments.json file.

    Returns
    -------
    list[dict]
        List of paired experiments. Each dict has keys:
        code, repeat, date, MAK (experiment dict), ANG (experiment dict).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group successful experiments by (code, repeat, date)
    groups = defaultdict(dict)
    for exp in data['experiments']:
        if exp['status'] != 'success':
            continue
        key = (exp['code'], exp['repeat'], exp['date'])
        view = exp['view']
        if view in ('MAK', 'ANG'):
            groups[key][view] = exp

    # Find pairs that have both MAK and ANG
    pairs = []
    for (code, repeat, date), views in groups.items():
        if 'MAK' in views and 'ANG' in views:
            pairs.append({
                'code': code,
                'repeat': repeat,
                'date': date,
                'MAK': views['MAK'],
                'ANG': views['ANG'],
            })

    # Sort for deterministic output
    pairs.sort(key=lambda p: (p['code'], p['repeat'], p['date']))
    return pairs


if __name__ == '__main__':
    import os

    # Default path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, 'processed_results', 'experiments.json')

    pairs = find_paired_experiments(json_path)
    print(f"Toplam MAK+ANG pair sayisi: {len(pairs)}")
    print(f"\nIlk 5 pair:")
    for p in pairs[:5]:
        print(f"  {p['code']} | {p['repeat']} | {p['date']} | "
              f"MAK hiz: {p['MAK']['metrics'].get('Tahmini Hiz', 'N/A')} | "
              f"ANG hiz: {p['ANG']['metrics'].get('Tahmini Hiz', 'N/A')}")
