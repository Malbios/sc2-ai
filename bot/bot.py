from sc2.bot_ai import BotAI
from sc2.data import Result


class CompetitiveBot(BotAI):
    """Main bot class that handles the game logic."""

    async def on_start(self):
        """Runs once at the start of the game."""
        print("Game started")

    async def on_step(self, iteration: int):
        """Runs continually throughout the game. Put your bot's logic here."""
        pass

    async def on_end(self, result: Result):
        """Runs once at the end of the game."""
        print(f"Game ended: {result}")
