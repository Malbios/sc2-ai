"""A roach group against marines and marauders (GroupFightTask).

Built for the headroom check: the attack, focus, threat and smart rules differ only in their
micro, so how far their results spread shows how much micro matters in this fight.
"""

import numpy as np

from tools.rl.examples.group_fight_task import (
    ATTACK_DANGEROUS,
    LIFE,
    GroupFightTask,
    Rule,
    closest_enemy_distance,
    life_kite_rule,
    step_away,
)

# The smart rule pulls a roach back below this share of its life while an enemy is this close.
PULL_BACK_BELOW = 0.35
PULL_BACK_DISTANCE = 7.0
LIFE_KITE_DISTANCES = (5.0, 7.0)  # center to center: a roach's range and a marauder's, plus radii


class RoachGroupTask(GroupFightTask):
    cooldown_scale = 32.0

    def baseline_policies(self) -> dict[str, Rule]:
        rules = {**super().baseline_policies(), "smart": _smart_rule}
        for distance in LIFE_KITE_DISTANCES:
            rules[f"life_kite_{distance:g}"] = lambda observation, d=distance: life_kite_rule(observation, d)
        return rules


def _smart_rule(observation: np.ndarray) -> int:
    """Shoot the most dangerous enemy in range; but a roach below PULL_BACK_BELOW life with an
    enemy within PULL_BACK_DISTANCE steps back, away from the enemies (closer ones count more)."""
    if observation[LIFE] < PULL_BACK_BELOW and closest_enemy_distance(observation) < PULL_BACK_DISTANCE:
        return step_away(observation)
    return ATTACK_DANGEROUS
