import unittest

from tools.rl.evaluate import summarize


class EvaluateTest(unittest.TestCase):
    def test_summary_rows(self):
        fights = [
            {"scenario": "x", "outcome": "win", "damage_dealt": 1.0, "damage_taken": 0.5, "game_seconds": 20},
            {"scenario": "x", "outcome": "timeout", "damage_dealt": 0.5, "damage_taken": 0.5, "game_seconds": 60},
        ]
        row = summarize(fights).splitlines()[1].split()
        self.assertEqual(row, ["x", "2", "50%", "0%", "50%", "75%", "50%", "40.0"])


if __name__ == "__main__":
    unittest.main()
