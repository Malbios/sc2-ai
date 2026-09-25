import unittest
from collections import Counter

from tools.batch.config import expand_games, parse_config
from tools.batch.summary import aggregate, format_comparison, win_rate

MINIMAL = {"maps": ["A", "B"], "opponents": [{"race": "Terran"}, {"race": "Zerg", "difficulty": "Hard"}]}


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        config = parse_config(MINIMAL)
        self.assertEqual(config.bot, "bot:CompetitiveBot")
        self.assertEqual(config.race, "Zerg")
        self.assertEqual(config.opponents[0].difficulty, "Easy")
        self.assertEqual(config.opponents[0].build, "RandomBuild")
        self.assertEqual(config.opponents[1].label(), "Zerg-Hard-RandomBuild")

    def test_rejects_unknown_names(self):
        for bad in ({"race": "Elf"}, {"race": "Zerg", "difficulty": "Impossible"}, {"race": "Zerg", "build": "Cheese"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_config({**MINIMAL, "opponents": [bad]})

    def test_rejects_empty_lists_and_bad_bot(self):
        with self.assertRaises(ValueError):
            parse_config({**MINIMAL, "maps": []})
        with self.assertRaises(ValueError):
            parse_config({**MINIMAL, "bot": "CompetitiveBot"})

    def test_expand_interleaves_matchups(self):
        games = expand_games(parse_config({**MINIMAL, "games_per_matchup": 2}))
        self.assertEqual(len(games), 8)
        # The first round covers every map x opponent once before any repeats.
        first_round = {(g.map, g.opponent.race) for g in games[:4]}
        self.assertEqual(len(first_round), 4)
        self.assertEqual(len({g.name() for g in games}), 8)


class SummaryTest(unittest.TestCase):
    RECORDS = [
        {"map": "A", "opponent": "T", "result": "Victory"},
        {"map": "A", "opponent": "Z", "result": "Defeat"},
        {"map": "B", "opponent": "T", "result": "Victory"},
        {"map": "B", "opponent": "Z", "result": "Crash"},
    ]

    def test_aggregate(self):
        by_opponent = aggregate(self.RECORDS, "opponent")
        self.assertEqual(by_opponent["T"]["Victory"], 2)
        self.assertEqual(by_opponent["Z"], Counter({"Defeat": 1, "Crash": 1}))
        self.assertEqual(sum(aggregate(self.RECORDS, None)["all"].values()), 4)

    def test_win_rate_counts_crashes_as_games(self):
        self.assertEqual(win_rate(aggregate(self.RECORDS, None)["all"]), 0.5)
        self.assertEqual(win_rate(Counter()), 0.0)

    def test_comparison_marks_missing_groups(self):
        before = aggregate(self.RECORDS, "map")
        after = aggregate(self.RECORDS[:2], "map")
        text = format_comparison("map", before, after)
        self.assertIn("+0pp", text)
        self.assertIn("only in one run", text)


if __name__ == "__main__":
    unittest.main()
