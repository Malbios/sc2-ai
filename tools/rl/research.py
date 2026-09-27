"""Researches the learner's upgrades (e.g. Burrow) once per game, before the first fight.

Spawned units have no research, so the learner spawns the buildings each upgrade needs in the
buildable spots farthest from the fights, and researches with fast build and free resources.
Upgrades belong to the player, so they last for the whole game.
"""

from sc2.bot_ai import BotAI
from sc2.dicts.unit_research_abilities import RESEARCH_INFO
from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

# Upgraded townhalls research everything the basic one does.
UPGRADED_FORMS = {
    UnitTypeId.HATCHERY: (UnitTypeId.LAIR, UnitTypeId.HIVE),
    UnitTypeId.COMMANDCENTER: (UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS),
}
BUILDING_SPACING = 8.0
TIMEOUT_SECONDS = 60.0


def research_buildings(upgrades: set[UpgradeId]) -> set[UnitTypeId]:
    """The buildings that research `upgrades`, plus the ones those researches require."""
    buildings = set()
    for upgrade in upgrades:
        researched_from = UPGRADE_RESEARCHED_FROM[upgrade]
        buildings.add(researched_from)
        required = RESEARCH_INFO[researched_from][upgrade].get("required_building")
        if required:
            buildings.add(required)
    for basic, upgraded in UPGRADED_FORMS.items():
        if basic in buildings and buildings & set(upgraded):
            buildings.discard(basic)
    return buildings


def building_spots(placement_grid, center: Point2, count: int, spacing: float = BUILDING_SPACING) -> list[Point2]:
    """`count` buildable cells, farthest from `center` first, at least `spacing` apart."""
    cells = [Point2((x, y)) for x in range(placement_grid.width) for y in range(placement_grid.height)
             if placement_grid[(x, y)]]
    spots = []
    for cell in sorted(cells, key=lambda cell: -cell.distance_to(center)):
        if all(cell.distance_to(spot) >= spacing for spot in spots):
            spots.append(cell)
            if len(spots) == count:
                return spots
    raise ValueError(f"the map has room for only {len(spots)} of {count} research buildings")


class UpgradeResearch:
    def __init__(self, upgrades: tuple[str, ...]):
        self.upgrades = {UpgradeId[name] for name in upgrades}
        self.started_loop: int | None = None
        self.ordered: set[UpgradeId] = set()

    async def done(self, bot: BotAI) -> bool:
        """True once every upgrade is researched; until then, moves the research along. Raises
        if it takes too long, which ends the game like a crash."""
        if not self.upgrades:
            return True
        missing = self.upgrades - bot.state.upgrades
        if not missing:
            return True
        if self.started_loop is None:
            await self._spawn_buildings(bot)
            self.started_loop = bot.state.game_loop
            return False
        if (bot.state.game_loop - self.started_loop) / 22.4 > TIMEOUT_SECONDS:
            raise RuntimeError(f"researching {sorted(u.name for u in missing)} took over {TIMEOUT_SECONDS:.0f} s")
        for upgrade in missing - self.ordered:
            researched_from = UPGRADE_RESEARCHED_FROM[upgrade]
            ability = RESEARCH_INFO[researched_from][upgrade]["ability"]
            buildings = bot.structures.of_type({researched_from, *UPGRADED_FORMS.get(researched_from, ())}).ready.idle
            if buildings and buildings.first(ability):
                self.ordered.add(upgrade)
        return False

    async def _spawn_buildings(self, bot: BotAI):
        await bot.client.debug_fast_build()
        await bot.client.debug_all_resources()
        buildings = sorted(research_buildings(self.upgrades), key=lambda building: building.name)
        spots = building_spots(bot.game_info.placement_grid, bot.game_info.map_center, len(buildings))
        await bot.client.debug_create_unit([(building, 1, spot, bot.player_id) for building, spot in zip(buildings, spots)])
