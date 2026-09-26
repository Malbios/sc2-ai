import unittest

from tools.rl.evaluate import summarize


class EvaluateTest(unittest.TestCase):
    def test_summary_rows(self):
        fights = [
            {"scenario": "x", "outcome": "win", "damage_dealt": 1.0, "damage_taken": 0.5, "game_seconds": 20,
             "kite_share": 0.8},
            {"scenario": "x", "outcome": "timeout", "damage_dealt": 0.5, "damage_taken": 0.5, "game_seconds": 60,
             "kite_share": 0.4},
        ]
        row = summarize(fights).splitlines()[1].split()
        self.assertEqual(row, ["x", "2", "50%", "0%", "50%", "75%", "50%", "40.0", "60%"])

    def test_fights_without_kite_share_count_as_zero(self):
        fights = [{"scenario": "x", "outcome": "loss", "damage_dealt": 0.0, "damage_taken": 1.0, "game_seconds": 5}]
        self.assertEqual(summarize(fights).splitlines()[1].split()[-1], "0%")


if __name__ == "__main__":
    unittest.main()
