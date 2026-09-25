# Linux machine setup (batch runs and training)

The tools in `tools/` run on a Linux machine with a native SC2 install. No Docker is involved:
python-sc2 starts and stops its own SC2 processes.

## Prerequisites

- **SC2 4.10 for Linux** (Base75689), the last Linux build Blizzard published and the one AI Arena
  runs. python-sc2 finds it at `~/StarCraftII`, or wherever `SC2PATH` points.
- **Maps** under `~/StarCraftII/Maps` (subfolders are fine):
  - the ladder maps the bot plays on (e.g. `2025PS2_Maps/*AIE*.SC2Map`)
  - `Melee/Flat64.SC2Map`, which RL training uses by default

  The Linux build looks for maps in a **lowercase** `maps` folder and fails with `InvalidMapPath`
  otherwise. If yours is called `Maps`, add a symlink: `ln -s Maps ~/StarCraftII/maps`.
- Python 3.10 or newer, with `venv`.
- `tmux`, so long runs survive a closed SSH session.

## Install

```
git clone https://github.com/Malbios/sc2-ai.git ~/sc2-ai
cd ~/sc2-ai
python3 -m venv venv
. venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-tools.txt
```

Install torch from the CPU-only index first. The default torch package includes the CUDA build,
which is several GB and useless without a GPU.

Update later with `git pull` (and rerun the second `pip install` if the requirements changed).

## Batch runs against the built-in AI

```
python -m tools.batch.run --config tools/batch/example.yaml [--parallel 4] [--games-per-matchup 3]
python -m tools.batch.summary runs/<run>                  # win rates of one run
python -m tools.batch.summary runs/<before> runs/<after>  # compare two runs
```

Each run gets a folder `runs/<UTC time>-<commit>[-dirty]/` with `results.jsonl`, `replays/`,
per-game `logs/` and `meta.json` (commit, dirty flag, config). "Dirty" means the bot had
uncommitted changes to tracked files, so the commit alone doesn't describe what played. The
replays open in sc2-observer.

## Long runs

Start a tmux session, run the command inside it, then detach with `Ctrl+b d`:

```
tmux new -s sc2
# ... run commands ...
tmux attach -t sc2      # reattach later
```

If something was killed hard, check for leftover SC2 processes with `pgrep -af SC2_x64` and stop
them with `pkill -f SC2_x64`. The `-f` matters: SC2's process name is `Main_Thread`, so a plain
`pgrep SC2_x64` never finds anything.
