"""A hydralisk group against zealots without Charge (GroupFightTask), for the headroom survey.

With Muscular Augments hydralisks are faster than zealots (3.94 against 3.15) and with Grooved
Spines they shoot from 6, so backing off while the weapon cools down should keep them out of
melee range; in a group, backing off also runs into allies and other zealots.

Needs the learner upgrades EVOLVEMUSCULARAUGMENTS and EVOLVEGROOVEDSPINES in the config.
"""

import numpy as np

from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    ATTACK_DANGEROUS,
    COOLDOWN,
    GroupFightTask,
    Rule,
    closest_enemy_distance,
    kite_rule,
    life_kite_rule,
    spread_rule,
    step_away,
)

KITE_DISTANCES = (2.0, 3.0, 5.0)
# Against marines a hydralisk fires from about 7 center to center, so a limit of 5 hardly ever
# lets it back off; these go past that, and ANY_DISTANCE is beyond any enemy it can see.
LONG_KITE_DISTANCES = (7.0, 9.0)
ANY_DISTANCE = 50.0
LIFE_KITE_DISTANCES = (3.0, 5.0)
SPREAD_SPACINGS = (1.0, 1.5)  # center to center, against splash such as a siege tank's
# Against mutalisks the search's weights back off at any distance only early in the cooldown,
# and always with a mutalisk within 5.
EARLY_KITE_DISTANCE = 5.0
EARLY_COOLDOWN_SHARES = (0.5, 0.7)


class HydraliskGroupTask(GroupFightTask):
    cooldown_scale = 13.0

    def baseline_policies(self) -> dict[str, Rule]:
        rules = {"attack": lambda observation: ATTACK_CLOSEST}
        for distance in KITE_DISTANCES:
            rules[f"kite_{distance:g}"] = lambda observation, d=distance: kite_rule(observation, d)
            rules[f"kite_threat_{distance:g}"] = lambda observation, d=distance: kite_rule(observation, d, ATTACK_DANGEROUS)
        for distance in LONG_KITE_DISTANCES:
            rules[f"kite_threat_{distance:g}"] = lambda observation, d=distance: kite_rule(observation, d, ATTACK_DANGEROUS)
        rules["kite_any"] = lambda observation: kite_rule(observation, ANY_DISTANCE)
        rules["kite_threat_any"] = lambda observation: kite_rule(observation, ANY_DISTANCE, ATTACK_DANGEROUS)
        for distance in LIFE_KITE_DISTANCES:
            rules[f"life_kite_{distance:g}"] = lambda observation, d=distance: life_kite_rule(observation, d)
        for spacing in SPREAD_SPACINGS:
            rules[f"spread_always_{spacing:g}"] = lambda observation, s=spacing: spread_rule(observation, s, always=True)
        for early in EARLY_COOLDOWN_SHARES:
            rules[f"early_kite_threat_{early:g}"] = lambda observation, e=early: early_kite_rule(
                observation, EARLY_KITE_DISTANCE, e, ATTACK_DANGEROUS)
        return rules


def early_kite_rule(observation: np.ndarray, distance: float, early: float, attack: int) -> int:
    """While the weapon cools down, back off from an enemy within `distance`, and from any enemy
    in sight while more than `early` of the cooldown is left (right after a shot); else `attack`.
    Coming back for the end of the cooldown keeps the next shot from being late."""
    closest = closest_enemy_distance(observation)
    cooling = observation[COOLDOWN] > 0
    if cooling and (closest <= distance or (closest < ANY_DISTANCE and observation[COOLDOWN] > early)):
        return step_away(observation)
    return attack
