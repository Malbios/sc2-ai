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
                                │ approved actions
                 ┌──────────────▼───────────────┐
                 │  Action Executor             │  dedupe, skip redundant orders,
                 └──────────────────────────────┘  issue python-sc2 calls
```

Rules that keep the layers honest:

- The World Model is the only thing that reads raw `BotAI` state (`self.units`, `self.structures`,
  `self.enemy_units`, ...). Everything else reads the World Model.
- Strategy and the managers never call action APIs. They return goals and intents.
- The Executor is the only layer that calls `self.train(...)`, `self.build(...)`,
  `unit.attack(...)` and ability usage.

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
  added one at a time.
- **Scouting:** keeps the World Model fresh, especially about enemy tech and expansions.

- **Input:** goals from Strategy, plus the World Model.
- **Output:** intents such as "train 2 drones", "build spawning pool near main", "squad A attack
  position P". Intents are requests; the Arbiter decides which ones happen.

## Arbiter / Resource Manager

Managers compete for the same minerals and the same units. Without an arbiter, a drone gets pulled
to build, mine and defend in the same frame. This layer takes over the old "Mastermind" job of
resolving conflicts, and gives "keeping the layers in sync" a concrete meaning.

- **Unit ownership:** every unit belongs to exactly one manager. Transfers are explicit (e.g.
  Production borrows a drone from Economy to build, then hands it back).
- **Resource reservation:** Production reserves the cost of a building when it commits to it, not
  when the build command lands, so other managers can't spend the same minerals meanwhile.
- **Priority:** when intents conflict, a fixed priority order decides (e.g. defense > supply >
  production > economy), and losing intents are dropped or deferred to the next step.
- **Input:** all managers' intents, plus the World Model.
- **Output:** the approved subset of intents, as actions for the Executor.

## Action Executor

Turns approved actions into python-sc2 calls.

- Skips orders a unit already has. Re-issuing the same command every step resets unit behavior
  and wastes actions.
- Merges duplicate actions from the same step.
- **Input:** approved actions from the Arbiter.
- **Output:** `self.train(...)`, `await self.build(...)`, `unit.attack(...)`, ability usage.

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
2. **Executor next.** The single place that turns actions into python-sc2 calls.
3. **Economy manager + data-driven build order.** A build order file that Strategy reads and an
   Economy/Production manager that returns intents. This is the thin vertical slice: economy, one
   build order, and "attack-move at supply 100".
4. **Arbiter** once a second manager needs workers or minerals.
5. **Army squads and micro controllers last**, one unit type at a time.

Each step should leave the bot fully playable. This is incremental, not a rewrite.
