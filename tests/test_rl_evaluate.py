import unittest

from tools.rl.evaluate import compare, summarize


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


class CompareTest(unittest.TestCase):
    def decision(self, situation, agree, outcome, scenario="x"):
        return {"scenario": scenario, "situation": situation, "agree": agree, "outcome": outcome}

    def test_rows_per_situation_with_win_and_loss_split(self):
        decisions = [
            self.decision("ready", True, "win"),
            self.decision("ready", False, "loss"),
            self.decision("ready", True, "loss"),
            self.decision("cooling", False, "win"),
        ]
        lines = compare(decisions).splitlines()
        self.assertEqual(lines[0], "x")
        self.assertEqual(lines[2].split(), ["cooling", "1", "25%", "0%", "0%", "-"])
        self.assertEqual(lines[3].split(), ["ready", "3", "75%", "67%", "100%", "50%"])

    def test_one_table_per_scenario(self):
        decisions = [self.decision("all", True, "win", "b"), self.decision("all", True, "timeout", "a")]
        tables = compare(decisions).split("\n\n")
        self.assertEqual([table.splitlines()[0] for table in tables], ["a", "b"])
        self.assertEqual(tables[0].splitlines()[2].split(), ["all", "1", "100%", "100%", "-", "-"])


if __name__ == "__main__":
    unittest.main()
