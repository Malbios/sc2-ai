"""Measures whether units kite: back away from the closest enemy while their weapon cools down.

Win rates alone can't tell kiting from chasing a fleeing enemy or standing and trading shots,
which is how earlier models looked successful without kiting at all.
"""

from sc2.unit import Unit
from sc2.units import Units

DANGER_MARGIN = 2.0  # beyond the enemy's reach, a gap still counts as "close enough to matter"
MIN_GAP_GROWTH = 0.05


class KiteMeter:
    """Counts the steps where a unit's weapon is cooling down with its closest enemy nearly in
    reach, and how many of those steps it opened the gap to that enemy."""

    def __init__(self):
        self._previous_gap: dict[tuple[int, int], float] = {}
        self.close_cooldown_steps = 0
        self.backing_off_steps = 0

    def update(self, unit: Unit, enemies: Units) -> None:
        if not enemies:
            return
        enemy = min(enemies, key=lambda other: unit.position.distance_to(other.position))
        gap = unit.position.distance_to(enemy.position)
        previous = self._previous_gap.get((unit.tag, enemy.tag))
        self._previous_gap[(unit.tag, enemy.tag)] = gap

        enemy_reach = enemy.ground_range + unit.radius + enemy.radius
        if previous is None or unit.weapon_cooldown <= 0 or gap > enemy_reach + DANGER_MARGIN:
            return
        self.close_cooldown_steps += 1
        if gap > previous + MIN_GAP_GROWTH:
            self.backing_off_steps += 1

    @property
    def share(self) -> float:
        """The share of close cooldown steps spent backing off: high when kiting, low when
        chasing or trading shots, 0 if the situation never came up."""
        return self.backing_off_steps / self.close_cooldown_steps if self.close_cooldown_steps else 0.0
