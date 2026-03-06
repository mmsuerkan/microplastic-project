"""
Tests for MAK+ANG pair finder.
"""
import sys
import os
import unittest

sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis_3d.pair_finder import find_paired_experiments

JSON_PATH = os.path.join(PROJECT_ROOT, 'processed_results', 'experiments.json')


class TestPairFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pairs = find_paired_experiments(JSON_PATH)

    def test_returns_at_least_300_pairs(self):
        """Pair sayisi en az 300 olmali."""
        self.assertGreaterEqual(
            len(self.pairs), 300,
            f"Beklenen >= 300 pair, bulunan: {len(self.pairs)}"
        )

    def test_each_pair_has_both_views(self):
        """Her pair'de hem MAK hem ANG olmali."""
        for pair in self.pairs:
            self.assertIn('MAK', pair, f"MAK eksik: {pair['code']}")
            self.assertIn('ANG', pair, f"ANG eksik: {pair['code']}")
            self.assertIsInstance(pair['MAK'], dict)
            self.assertIsInstance(pair['ANG'], dict)

    def test_mak_ang_have_matching_keys(self):
        """MAK ve ANG ayni code, repeat, date'e sahip olmali."""
        for pair in self.pairs:
            mak = pair['MAK']
            ang = pair['ANG']
            self.assertEqual(
                mak['code'], ang['code'],
                f"Code uyumsuz: MAK={mak['code']}, ANG={ang['code']}"
            )
            self.assertEqual(
                mak['repeat'], ang['repeat'],
                f"Repeat uyumsuz: MAK={mak['repeat']}, ANG={ang['repeat']}"
            )
            self.assertEqual(
                mak['date'], ang['date'],
                f"Date uyumsuz: MAK={mak['date']}, ANG={ang['date']}"
            )

    def test_views_are_correct(self):
        """MAK experiment view='MAK', ANG experiment view='ANG' olmali."""
        for pair in self.pairs:
            self.assertEqual(pair['MAK']['view'], 'MAK')
            self.assertEqual(pair['ANG']['view'], 'ANG')

    def test_pair_keys_match_experiment(self):
        """Pair dict'teki code/repeat/date, experiment datalariyla eslesmeli."""
        for pair in self.pairs:
            self.assertEqual(pair['code'], pair['MAK']['code'])
            self.assertEqual(pair['repeat'], pair['MAK']['repeat'])
            self.assertEqual(pair['date'], pair['MAK']['date'])


if __name__ == '__main__':
    unittest.main()
