# Roadmap

What's next, in order. Details and reasons are in `starcraft-bot-design.md`; this file is the
checklist. Tick items off as they're done.

The current goal is to find out whether RL micro beats scripted micro. The bot comes after that.

## 1. Training toolkit, for single-unit models

- [x] Research and upgrades for spawned units (`learner.upgrades`, any upgrade python-sc2 knows)
- [x] Actions that cast abilities: burrow and unburrow
- [ ] Actions that cast abilities: bile at a point (a target's position plus a lead in one of
      several directions)
- [x] Enemy detection: an Overseer that stays with its army (`enemy_support`), and tasks that
      see how close detection is
- [x] Start a fresh SC2 game before SC2's 6.5-game-hour limit
- [ ] Enemies that move while fighting: a scripted enemy that kites, and one that walks out of
      bile
- [ ] Enemy unit types the bot will really face (marines, marauders, zealots, stalkers,
      zerglings, roaches), not only roaches
- [ ] Check each new task against the built-in AI too, as an enemy the model never trained on

## 2. Models, one at a time

Each: a scripted baseline first, then train, then evaluate (200 fights, per checkpoint) against
chasing, leashing and built-in AI enemies. It counts as a win for RL only if it beats its script.

- [ ] Ravager (next): bile aiming against moving enemies (baseline: bile where the target is now)
- Roach, parked: the burrow rule already wins everything without detection, and nothing wins
  with an Overseer, so there's no room to beat it (see the design doc's roach lessons)
- [ ] Hydralisk: focus fire and kiting in groups (needs the group items below)

## 3. Training toolkit, for groups

Several learner units already share one model in a fight, each deciding for itself. Missing:

- [ ] Allies in the observation (where they are, their life), so units can react to each other
- [ ] More than the 2 closest enemies in the observation
- [ ] A reward that tells which unit did well; today every unit gets the team's reward
- [ ] Faster training when a fight has many units (each unit's decision is one training step)
- [ ] Then possibly banelings, mutalisks, zergling surrounds

## 4. The bot itself

The design's build order puts learned micro last: a model only helps once the bot can play a full
game and has a controller slot to put it in. Today the bot only builds drones, a spawning pool
and zerglings.

- [ ] World Model: one world-state object per step that everything else reads
- [ ] Economy/Production manager with a data-driven build order (roach-ling), attack-move at a
      set supply
- [ ] Arbiter, once a second manager competes for workers or minerals
- [ ] Army squads and scripted micro controllers, one unit type at a time: roach, ravager,
      hydralisk. These are the fallback wherever a model doesn't beat them.
- [ ] Shared observation builder, used by both the bot and training, so models see the same
      inputs in real games as in training

## Parked

- Stalker against the built-in AI: 94% wins, 61-second fights (the smart rule wins 100%). Watch
  `models/tracking-kite-gamma/eval-final-builtin.SC2Replay` if it becomes relevant again.
