# sc2-ai

A StarCraft II bot built on [python-sc2](https://github.com/BurnySc2/python-sc2) (installed via
the `burnysc2` package).

## Two-machine setup

The bot logic (this Python process) runs on this machine. The actual SC2 game client runs on a
separate machine on the same LAN.

### 1. On the remote machine (the one with the SC2 game window)

Launch SC2 listening on the network instead of just localhost:

```
SC2_x64.exe -listen 0.0.0.0 -port 5001
```

Make sure the map files listed in `config.py`'s `MAP_POOL` exist in that machine's `Maps` folder,
and that its firewall allows inbound connections on the port above.

### 2. On this machine

1. Set up a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Edit `config.py` and set `REMOTE_HOST` to the remote machine's LAN IP (and `REMOTE_PORT` if you
   changed the port above).
3. Run the bot:
   ```
   python run.py
   ```

`helper-scripts/ping_sc2.py <REMOTE_HOST>` is useful to sanity-check connectivity to the remote
SC2 instance before running the bot.

## Project layout

- `bot/bot.py` — the bot's logic (`CompetitiveBot`, a `BotAI` subclass). `on_step` is where most
  of your logic goes.
- `config.py` — bot name/race, remote host/port, map pool, opponent settings.
- `run.py` — connects to the remote SC2 instance, creates a game, and runs the bot.
