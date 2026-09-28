# Roadmap

What's next, in order. Details and reasons are in `starcraft-bot-design.md`; this file is the
checklist. Tick items off as they're done.

The current goal is to find out whether RL micro beats scripted micro. The bot comes after that.

## 1. Training toolkit, for single-unit models

- [x] Research and upgrades for spawned units (`learner.upgrades`, any upgrade python-sc2 knows)
- [x] Actions that cast abilities: burrow and unburrow
- [x] Actions that cast abilities: bile at one of the 2 closest enemies with a lead along its
      movement; tasks can see which abilities are ready
- [x] A scripted enemy that steps out of bile (with a reaction time)
- [x] Enemy detection: an Overseer that stays with its army (`enemy_support`), and tasks that
      see how close detection is
- [x] Start a fresh SC2 game before SC2's 6.5-game-hour limit
- [x] Enemies that move while fighting: a scripted enemy that kites (`enemy_behavior: kite`)
- [ ] One model per own unit type that fights any enemy type (not one model per matchup): inputs
      that describe each seen enemy by its properties (its range and speed compared with mine,
      its damage against me, whether it can hit me), more than the 2 closest enemies, and enemy
      types drawn at random per fight. First test after the ravager run: add marines as a second
      enemy type and check whether one model handles both. Then the types the bot will really
      face (marines, marauders, zealots, stalkers, zerglings, roaches).
- [ ] Check each new task against the built-in AI too, as an enemy the model never trained on

## 2. Models, one at a time

Each: a scripted baseline first, then train, then evaluate (200 fights, per checkpoint) against
chasing, leashing and built-in AI enemies. It counts as a win for RL only if it beats its script.

- [ ] Ravager (next): bile aiming against moving enemies. Baselines measured; the bar per enemy is
      the best rule (see the design doc's ravager table). A first run on 1 vs 2 roaches only
      plateaued below every rule; with winnable 1 vs 1 fights mixed in it still didn't learn to
      aim bile (stopped at 500k). Paying for damage and masking bile actions got it level with
      the lead rule against 1 chaser at 500k, but later checkpoints traded that for the
      leashing fights; no checkpoint beats the best rule anywhere. Kept for later: a last test
      where the lead rule aims and the model only decides when to bile, which enemy, and moves.
- Roach, parked: the burrow rule already wins everything without detection, and nothing wins
  with an Overseer, so there's no room to beat it (see the design doc's roach lessons)
- [ ] Hydralisk: focus fire and kiting in groups (needs the group items below)

## 3. Training toolkit, for groups

Several learner units already share one model in a fight, each deciding for itself. Missing:

- [x] Allies in the observation (where they are, their life), so units can react to each other
      (ZerglingSurroundTask: the 4 closest allies)
- [x] More than the 2 closest enemies in the observation (ZerglingSurroundTask: 4)
- [x] A reward that tells which unit did well (`MicroTask.share_team_reward`; zerglings: damage
      goes to the ones that attacked)
- [x] Faster training when a fight has many units: group tasks decide for all units at once, one
      training stream per unit (`MicroTask.group_slots`)
- [ ] Zergling surround against marines (now): baselines measured, bar per fight is attack-move.
      A first run from scratch never learned to engage (stopped at 6.85M slot decisions). With
      an attack-biased start, training only on living units and per-unit reward, the best
      checkpoints played like attack-move, then training collapsed.
- [ ] Then possibly banelings, mutalisks

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
