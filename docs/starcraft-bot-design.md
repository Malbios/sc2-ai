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
- Reward is roughly damage dealt minus damage taken, plus a bonus for winning the fight.
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

What the first milestone taught us (stalker kiting, 2026-09):

- **Pay for the behavior, keep the task small.** A model with 5 actions, 52 inputs and a reward of
  damage dealt minus damage taken never won against 2 roaches: the damage-taken penalty taught it
  to run away. A port of sharknice's SharkyRLMatrixTraining task (2 actions: attack or retreat; 3
  inputs: weapon cooldown, distance, distance change; +1 per step with the weapon cooling down,
  life left as the win reward, no penalties) learned flawless 1v1 kiting in 100k decisions.
- **Train on the easy fight, then transfer.** The model trained only on 1 stalker vs 1 roach won
  100% of 1 stalker vs 2 roaches fights. Training on the hard fight from scratch never won once.
- **Aggressive learning settings can lock in early.** At learning rate 0.01 with 16 epochs per
  batch, the model stopped exploring within about 5,000 decisions and kept whatever it did then:
  kiting in 1v1 (lucky), running away in 1v2. Refining an existing model uses gentle settings.
- **Measure against baselines and distrust sudden wins.** Every model is compared with random
  actions and a hand-written rule. A 97% win rate once turned out to be a bug (enemies out of
  sight counted as dead), so winning by running away looked like winning.

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
