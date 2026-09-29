"""A hydralisk group against zealots without Charge (GroupFightTask), for the headroom survey.

With Muscular Augments hydralisks are faster than zealots (3.94 against 3.15) and with Grooved
Spines they shoot from 6, so backing off while the weapon cools down should keep them out of
melee range; in a group, backing off also runs into allies and other zealots.

Needs the learner upgrades EVOLVEMUSCULARAUGMENTS and EVOLVEGROOVEDSPINES in the config.
"""

from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    ATTACK_DANGEROUS,
    GroupFightTask,
    Rule,
    kite_rule,
)

KITE_DISTANCES = (2.0, 3.0, 5.0)


class HydraliskGroupTask(GroupFightTask):
    cooldown_scale = 13.0

    def baseline_policies(self) -> dict[str, Rule]:
        rules = {"attack": lambda observation: ATTACK_CLOSEST}
        for distance in KITE_DISTANCES:
            rules[f"kite_{distance:g}"] = lambda observation, d=distance: kite_rule(observation, d)
            rules[f"kite_threat_{distance:g}"] = lambda observation, d=distance: kite_rule(observation, d, ATTACK_DANGEROUS)
        return rules
