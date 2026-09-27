# StarCraft II bot architecture

## Concept

A bot that reads game state, decides what to do, and issues actions all in one place works while
it only knows a handful of rules, but it doesn't scale. Every new rule has to know about every
other rule's state, two rules can grab the same worker or spend the same minerals in one frame,
and there's no single place that could answer "what is this bot actually trying to do right now?"

The design below splits that work into layers that run in a fixed order once per game step.
Each layer only talks to the one next to it, and decisions get more concrete as they move down:
perception, then goals, then intents, then approved actions, then API calls. This doc records the
intent and target shape; it is reached incrementally (see "Suggested build order").

## Data flow, once per step

```
                 ┌──────────────────────────────┐
                 │  Game Interface (transport)  │  python-sc2: connect, step loop,
                 └──────────────┬───────────────┘  protobuf in/out
                                │ raw observation
                 ┌──────────────▼───────────────┐
                 │  World Model (perception)    │  typed units, memory of enemies
                 │  + Map Analysis (static)     │  in fog, clusters, relative strength
                 └──────────────┬───────────────┘
                                │ read-only state
                 ┌──────────────▼───────────────┐
                 │  Strategy                    │  "what are we trying to do":
                 └──────────────┬───────────────┘  build order, army mix, posture
                                │ goals
    ┌───────────┬───────────────┼──────────────┬─────────────┐
    ▼           ▼               ▼              ▼             ▼
 Economy    Production       Army/Tactics    Scouting     (Tech, Supply...)
 (workers)  (build/train)    (squads→micro)
    └───────────┴───────────────┼──────────────┴─────────────┘
                                │ intents (requests, not commands)
                 ┌──────────────▼───────────────┐
                 │  Arbiter / Resource Manager  │  unit ownership, mineral/gas
                 └──────────────┬───────────────┘  reservations, priority
                                │
                                ▼ python-sc2 unit commands
                                  (batched by python-sc2 per step)
```

Rules that keep the layers honest:

- The World Model is the only thing that reads raw `BotAI` state (`self.units`, `self.structures`,
  `self.enemy_units`, ...). Everything else reads the World Model.
- Strategy and the managers never call action APIs. They return goals and intents.
- The Arbiter is the only layer that issues python-sc2 commands (`unit.attack(...)`,
  `unit.move(...)`, `worker.build(...)`, `self.train(...)`, ability usage). Never call
  `self.client.actions(...)` directly, since that bypasses python-sc2's per-step batch.

## Game Interface

Connects to SC2, advances the game, and turns observations into state and actions into requests.

- **Input / output:** raw protobuf on the wire; `BotAI` state and action calls on the Python side.
- **Notes:** step mode is deterministic and is what ladders use, so it's the right default for
  testing. Realtime mode is for playing against humans. The two-host LAN setup is the one unusual
  part of this layer; keep connection and host/port setup separate from bot logic.

## World Model + Map Analysis

Turns raw observation into something the other layers can use without each of them re-deriving
it from `self.units`/`self.enemy_units`/`self.structures` on its own. Raw observations are
stateless, but the bot needs memory, so this layer is where state persists within a game.

- **Dynamic state (every step):**
  - own units, structures, economy, tech, and pending orders (e.g. "a drone is already on its
    way to build a spawning pool, don't send a second one")
  - enemy units and structures, including ones last seen in fog of war, with when and where
  - clusters of units, and an estimate of enemy strength per cluster
  - relative strength (army value, economy, tech) so Strategy can ask "am I on the losing side?"
  - a history of what has happened this game (timings seen, attacks, losses) that Strategy can
    categorize
- **Static map analysis (once at game start):** expansion locations, regions, chokes, ramps,
  pathing.
- **Input:** `BotAI` state each step.
- **Output:** a world-state object that Strategy, managers and the Arbiter read. Other layers
  never write to it.
- **Open questions:** clustering and strength estimation need actual algorithms; nothing like that
  exists yet, so this layer has the most net-new logic. python-sc2 already covers expansion
  locations and basic pathing; chokes and regions may need a map-analysis library.

## Strategy

A small, slow-changing layer that picks a plan for the current game state and expresses it as
goals, e.g. "2-base, target roach-ling, expand at 3:00, defend until 8 roaches."

- Where am I in the tech tree?
- What has my opponent been doing this game?
- Am I on the losing side?
- Should I defend or push?

- **Input:** the World Model.
- **Output:** goals for the managers (target composition, economy targets, posture, timings).
- **Build orders:** keep them data-driven (YAML or JSON) so they can change without code changes.
- **Out of scope:** remembering opponent behavior across games. Opponent play varies too much from
  game to game for it to be reliable, so every game starts fresh.

## Managers

Domain specialists that turn Strategy's goals into intents. Each one only reasons about its own
domain.

- **Economy:** worker saturation, gas, transfers between bases, long-distance mining.
- **Production:** building placement, training, upgrades, supply (overlords).
- **Army / Tactics:** groups units into squads and gives each squad a task (defend, attack,
  harass). Per-unit control is delegated to micro controllers.
- **Micro controllers:** the long-term goal is that for every ability of every unit there's an
  explicit instruction on how to use it well: when to burrow, when to kite, when to focus-fire,
  when to split against splash. Controllers are per unit type (or per ability), so they can be
  added one at a time. Controllers also decide when an order should be re-issued: skip
  near-identical orders a unit is already carrying out, since re-issuing resets unit behavior
  (kiting is the exception that needs frequent re-orders).
- **Scouting:** keeps the World Model fresh, especially about enemy tech and expansions.

- **Input:** goals from Strategy, plus the World Model.
- **Output:** intents such as "train 2 drones", "build spawning pool near main", "squad A attack
  position P". Intents are requests; the Arbiter decides which ones happen.

## Arbiter / Resource Manager

Managers compete for the same minerals and the same units. Without an arbiter, a drone gets pulled
to build, mine and defend in the same frame. This layer resolves those conflicts.

- **Unit ownership:** every unit belongs to exactly one manager. Transfers are explicit (e.g.
  Production borrows a drone from Economy to build, then hands it back).
- **Resource reservation:** Production reserves the cost of a building when it commits to it, not
  when the build command lands, so other managers can't spend the same minerals meanwhile.
- **Priority:** when intents conflict, a fixed priority order decides (e.g. defense > supply >
  production > economy), and losing intents are dropped or deferred to the next step.
- **Input:** all managers' intents, plus the World Model.
- **Output:** the approved subset of intents, issued as python-sc2 commands. python-sc2 already
  batches them into one request per step, merges identical orders, and skips orders that exactly
  match what a unit is already doing.

## Learned micro controllers

Some micro controllers can be small models trained with reinforcement learning instead of
hand-written rules. They plug into the existing micro controller slot; nothing above that layer
knows or cares whether a controller is scripted or learned.

- **One controller interface.** A controller takes its units, the squad's task and the World Model,
  and returns orders. Scripted and learned controllers for the same unit type are interchangeable.
- **One model per own unit type** ("zergling model", "roach model", ...). Each unit decides for
  itself based on its surroundings, so one model handles any composition. Unit-specific abilities
  stay in that type's own action set.
- **Scripted first.** Every learned controller has a scripted counterpart for its unit type. It is
  the baseline the model has to beat and the fallback when the model is missing or misbehaves.
- **Only where micro pays off.** Each unit type's model is an optional experiment, not a required
  layer. Train models for unit types where control makes a big difference (e.g. banelings,
  mutalisks, roaches with burrow, ravagers, queens), not for ones a script handles well (e.g.
  zerglings). A model can also cover a single decision (target choice, retreat or not) with the
  script doing the rest. If a model never beats its script, it doesn't ship.
- **Shared observation builder.** A fixed-size, unit-centered observation built from the World
  Model (e.g. nearest 8 enemies and allies with relative position, HP, shields, weapon cooldown,
  plus the squad's target direction). Training and live play use the exact same function, so the
  model never sees different inputs in real games than it was trained on. It must contain whatever
  tells situations apart: unit types, relevant buffs (e.g. stim), and a view range that covers the
  longest-ranged threats (a sieged tank hits from 13).
- **Discrete actions.** The model picks from a small set (attack nearest, attack weakest, retreat,
  move toward squad target, hold, unit-specific abilities like burrow). The controller maps the
  choice to python-sc2 commands. This keeps models tiny and their output limited to orders the
  rest of the bot understands.
- **Ownership and output unchanged.** The model only commands units its squad owns, and its orders
  go out through the same path as every other order.
- **Frame budget.** Run inference every few steps, and only for units in or near combat.

Training lives outside the bot:

- A separate training setup in this repo runs SC2 natively on a Linux machine (no Docker), in step
  mode, using python-sc2 and the same observation builder, action mapping and controller as the
  bot. The model decides every few game frames (python-sc2's `game_step`), not every frame.
- Learning method: PPO from Stable-Baselines3, with environments in the Gymnasium format.
- Reward is set per task (damage traded, a bonus for firing, a bonus for winning); see the lessons
  below for what each one taught the model.
- Match the discount factor (gamma) to the fight length: at 0.99 a reward half a minute away
  counts almost nothing, so a reward paid only at the end of the fight barely counts. Long fights
  need 0.999.
- **Randomized scenarios.** Each training round samples a scenario (unit types, counts, uneven
  fights, start positions) from a weighted list. The weights decide whether training goes step by
  step (1v1 first) or mixed from the start; that is a setting to try, not a fixed choice. Easy
  scenarios stay in the mix so the model doesn't forget them.
- **Enemy control**, all three available from the start:
  - built-in AI
  - a simple scripted enemy (predictable, for clear small scenarios)
  - a frozen copy of a model (self-play: one side learns while the other stays fixed, and they swap
    periodically). Needs one SC2 client per side, so fewer games run in parallel.
- **Scores per scenario**, not one combined score: win rate and damage traded for each scenario
  separately, so it shows when learning one scenario makes another worse.
- The bot loads frozen weights at startup and does not learn during games.
- **First milestone:** reproduce a known result ("1 stalker vs 2 roaches") to prove the tooling
  works before training real Zerg models.

What the first milestone taught us (stalker kiting, 2026-09). 1 stalker vs 1 roach: learned. 1 vs 2
roaches: a model with more inputs and actions beat both the kite rule (39%) and the smarter
hand-written rule (82%), winning 98%, and later 100% against roaches that chase and 100% with 3%
life lost against roaches that give up.

Why sharknice's setup (SharkyRLMatrixTraining) learns kiting, and ours first didn't. Two causes,
each confirmed by changing only that one thing (his task on his map, 1 stalker vs 1 roach). His task
has 3 inputs about the closest enemy (weapon cooldown, distance, distance change), 2 actions (attack
closest, run from closest) and pays for every decision the weapon is cooling down, i.e. for firing
often.

- **His enemy leashes.** It is the built-in AI, whose units defend their start location: they
  chase, give up at a leash distance and walk back home. On his map both start locations are at
  the center, so the roach turns around near the map edge and the stalker gets free shots. Even
  clumsy kiting pays from the first fights. Trained that way, the model kited with 5% life lost.
  (On a map with corner start locations, the same leash makes enemy units walk to their corner,
  which looks like fleeing.)
- **His learning rate locks in early.** At 0.01 with 16 epochs per batch, exploration died within
  5,000 to 20,000 decisions and the model kept whatever it did then. Against a roach that never
  gives up, attacking non-stop also wins 1v1, so it locked onto attack-only. At 0.0003 it kept
  exploring and learned kiting against that roach directly.

Evaluating:

- **Judge by 200 fights, not 30.** With 30 fights, a win rate is only accurate to about ±18
  points: one model scored 37% and then 57% with identical settings. 200 fights (about ±7) take
  only a few minutes.
- **Compare with a hand-written rule.** Every task gets baseline policies (attack only, retreat
  only, the kite rule: attack when the weapon is ready, else run). A model is only interesting if
  it beats the best rule. `evaluate --compare-with` shows, per situation, how often a model picks
  the rule's action, and whether its disagreements happen in won or lost fights.
- **Check that the enemy actually fights.** Retreat-only must lose (or time out against a leashing
  enemy). A model that "won" 100% against built-in AI roaches was chasing them home, with zero
  kiting in the replay. Micro training uses the scripted enemy (it always attacks the closest
  unit) unless leashing is the point.
- **Distrust sudden wins and watch a replay.** Fake results so far: a 97% win rate from enemies
  out of sight counted as dead, and 100% from enemies walking home.
- **Measure the behavior, not only the outcome.** The kite share (how often a unit backs off while
  its weapon cools down with an enemy close) told chasing from kiting when win rates could not.

Training:

- **At learning rate 0.01, keep checkpoints.** A model that kited well for most of its run flipped
  to "always retreat" in its last ~3,000 decisions, which is what got saved as final.
- **Refine with gentle settings.** A locked-in model barely changes when refined: it has stopped
  exploring and can only polish what it does.
- **A damage-taken penalty teaches running away.** With 5 actions, 52 inputs and a reward of
  damage dealt minus damage taken, the model never won against 2 roaches and learned to flee.
- **The inputs and actions set the ceiling, not the training.** Against 2 roaches, both models
  (from either learning rate, with a win reward of 1 or 100) chose the kite rule's action in 90
  to 100% of decisions, and won 41% vs the rule's 37% (200 fights each, a tie). The model sees only
  the closest roach and can only run straight away from it; it doesn't know where the second roach
  or the map edge is. Given that, the rule is already the best it can do. Beating it needs inputs
  and actions a rule doesn't use well.
- **With more inputs and actions, training beat the kite rule.** FreeKiteTask adds the second
  roach, the free distance in 8 directions, "attack weakest" and 8 run directions (same reward).
  Against 2 roaches (200 fights each): the kite rule 39%, the model 52% after 500k decisions, 74%
  after 1M and 98% after 1.5M (90% at 1.65M). A hand-written rule using the same extras (attack
  the weakest, run away from both roaches toward open ground) wins 82%, so richer tasks need a
  smarter rule as the bar too. The model plays differently from that rule: it spends about two
  thirds of its decisions near a wall, where the rule would not be.
- **Evaluate checkpoints, not only the final model.** Training win rates swing by 20 points
  between batches, and the best model is often an earlier checkpoint. Training on to 2M decisions
  gave checkpoints between 76% and 90%; none matched the 98% at 1.5M.
- **Training win rates run far below evaluated ones.** During training the model samples its
  actions to keep exploring; evaluation always takes its most likely action. 85% in training was
  98% evaluated, 60% was 74%.
- **A model can be tuned to its training enemy.** Against the built-in AI's roaches (which give
  up and walk home) the 98% model won everything but lost 30% of its life, while the kite and
  smart rules lost none. It trades life for speed, which only pays against enemies that never stop
  chasing. Keep the script as the fallback where it plays better.
- **Know where a model's limit is.** Against 3 roaches, everything lost: the 98% model won 0% (it
  only sees the 2 closest), the smart rule 2%, the kite rule 0%.
- **Training takes much longer when the task is richer.** From scratch with 10 actions, the model
  won its 1-roach fights early, but no 2-roach fight until ~150k decisions, although both were
  mixed in from the start (75% 2 roaches, 25% 1 roach). Then it climbed slowly.
- **An easy enemy in the mix can keep the hard skill from being learned.** TrackingKiteTask
  (FreeKiteTask plus how far each seen roach moved since the last decision) trained 2M decisions
  from scratch against chasing and leashing roaches in equal parts. Against leashing roaches it won
  90% in training within 250k decisions by trading hits until they walked away. Against chasing
  roaches it stayed at 0 to 3% in training for the whole run: it never found kiting. Evaluated (200
  fights each, checkpoints from 1M on) it won 2 to 22% against 2 chasing roaches, and lost 25 to
  37% of its life against 2 leashing ones, worse than the 98% model (11%) and the rules (0%).
- **Warm-start from a model that already has the hard skill.** `train --warm-start` copies a
  model's weights into a task with more inputs at the end, with zero weight on the new ones, so it
  first plays exactly like the old model. Started from the 98% model and trained 1M decisions on
  the same mix (learning rate 0.0001), it kept the kite: 92 to 98% against 2 chasing roaches at
  every checkpoint (98% at 1M). But it did not learn to play safe against quitters: life lost
  against 2 leashing roaches went from 12% (250k) to 21% (1M), and against the built-in AI it lost
  33% (the 98% model 30%).
- **The discount factor can make taking damage the better deal.** With gamma 0.99 a reward counts
  1% less per decision, about a third after 13 seconds, and life left is only paid in the win
  reward at the end. Against leashing roaches the smart rule's safe win takes 48 seconds (worth
  about 0.99^358 x 100 = 3 at the start), the model's trade 28 seconds with 21% lost (about
  0.99^209 x 79 = 10). The model does what it is paid for.
- **With gamma 0.999 the model learned to play safe.** Warm-started from the gamma 0.99 run's model
  and trained 1M decisions more (only gamma changed), life lost against 2 leashing roaches fell
  from 10% (250k) to 2 to 3% (750k and on), close to the rules' 0%. Against 2 chasing roaches it
  kept winning (100% at 750k and final) and lost less life too (40 to 48% instead of about 75%),
  in longer fights (37 to 46 seconds instead of 27). Against the built-in AI (never trained on)
  the final model lost 23% of its life instead of 30%, but won only 94% (4% losses, 2% timeouts,
  61-second fights), and the 750k checkpoint only 82%. It waits too long against an enemy it has
  not seen; the smart rule still wins 100% there with no damage.

Tooling pitfalls:

- `debug_show_map` is one game-wide toggle, so only one client may send it; two clients sending it
  switch it off again.
- sharknice's MicroTraining map turns fog off in its map script. Combined with `debug_show_map`,
  enemy units became untargetable snapshots. MicroTraining410 is a 4.10 port without that script.
- An SC2 game can crash mid-fight in long runs ("not in a game"). The driver must end that fight
  and restart the game; raising instead hung the whole training. Stop the two clients of a game
  together, and cancel each only once: python-sc2 exits the process when a request is cancelled
  twice.
- python-sc2 details: count enemies as dead only via `state.dead_units`, and save
  replays only while the learner is waiting for its next step.

## Cross-cutting concerns

- **Frame budget.** Expensive work (pathing, clustering, influence maps) runs every N steps or is
  spread across steps, not every step.
- **Debug drawing.** SC2 can draw text, spheres and lines in-game. Use it from the start to show
  squad targets, reserved building spots and threat levels.
- **Testability.** If Strategy and the managers are close to pure functions of the World Model,
  they can be unit-tested against saved observations without launching SC2.
- **Replays and decision logs.** Log every Strategy decision with a game timestamp. Watching a
  replay next to the log is the main debugging tool.

## Suggested build order

Build the layers one at a time without ever leaving the bot unable to play a full game.

1. **World Model first.** A world-state object built once per step from `BotAI` state, which
   everything else reads instead of raw state.
2. **Economy manager + data-driven build order.** A build order file that Strategy reads and an
   Economy/Production manager. This is the thin vertical slice: economy, one build order, and
   "attack-move at supply 100". Until the Arbiter exists, this manager issues python-sc2 commands
   directly.
3. **Arbiter** once a second manager needs workers or minerals. It takes over issuing commands, and
   managers switch to returning intents.
4. **Army squads and scripted micro controllers**, one unit type at a time.
5. **Learned micro controllers** last, one unit type at a time, each replacing a scripted
   controller only once it beats it in the same scenarios.

Each step should leave the bot fully playable. This is incremental, not a rewrite.
