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


EVALUATE_LOG = """\
2026-09-28 10:00:00.000 | INFO     | sc2.main:_host_game:228 - Status.launched
ravager_1v1_chase 1/2: win
SC2 game crashed, restarting on the next reset:
ravager_1v1_chase 2/2: win

scenario           fights   win  loss   tie  dealt  taken  seconds  kite
ravager_1v1_chase       2  100%    0%    0%   100%    50%     10.4   43%

2026-09-28 10:01:00.000 | INFO     | sc2.sc2process:_close_connection:216 - Closing connection
"""


class EvaluateManyTest(unittest.TestCase):
    def test_summary_table_is_cut_from_the_log(self):
        from tools.rl.evaluate_many import summary_table

        self.assertEqual(summary_table(EVALUATE_LOG).splitlines(), [
            "scenario           fights   win  loss   tie  dealt  taken  seconds  kite",
            "ravager_1v1_chase       2  100%    0%    0%   100%    50%     10.4   43%",
        ])

    def test_log_without_a_table(self):
        from tools.rl.evaluate_many import summary_table

        self.assertEqual(summary_table("ravager_1v1_chase 1/2: win\n"), "")

    def test_crashes_are_counted(self):
        from tools.rl.evaluate_many import crash_count

        self.assertEqual(crash_count(EVALUATE_LOG), 1)

    def test_models_are_named_by_their_path_in_the_repo(self):
        from tools.batch.run import REPO_ROOT
        from tools.rl.evaluate_many import display_name

        self.assertEqual(display_name(str(REPO_ROOT / "models" / "run-a" / "final.zip")), "models/run-a/final.zip")

    def test_evaluate_command(self):
        from tools.rl.evaluate_many import evaluate_command

        command = evaluate_command("c.yaml", "m.zip", 200, "builtin")
        self.assertEqual(command[1:], ["-m", "tools.rl.evaluate", "--config", "c.yaml", "--model", "m.zip",
                                       "--episodes", "200", "--enemy-mode", "builtin"])
        self.assertNotIn("--enemy-mode", evaluate_command("c.yaml", "m.zip", 200, None))
        self.assertEqual(evaluate_command("c.yaml", "m.zip", 200, None, stochastic=True)[-1], "--stochastic")
        self.assertNotIn("--stochastic", command)


if __name__ == "__main__":
    unittest.main()
