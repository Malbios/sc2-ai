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
  layer. Train models where one unit's own decisions make a big difference and scripts are crude:
  roaches (kiting, when to burrow and heal), ravagers (aiming bile where a moving target will be)
  and hydralisks (focus fire and kiting). Skip ones a simple script handles well (queens'
  transfuse, zerglings). Micro that is a group effort (baneling connects and splits, mutalisk
  stacking, zergling surrounds) needs models whose units see and coordinate with each other, which
  comes later; a timed script already does as well as a trained model in the SMAC baneling fight.
  A model can also cover a single decision (target choice, retreat or not) with the script doing
  the rest. If a model never beats its script, it doesn't ship.
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

What the roach attempt taught us (burrow micro, 2026-09). No model was trained: the baselines
showed the task can't tell whether RL beats a rule.

- **Research for spawned units:** spawn the research buildings and research with fast build and
  free resources (`learner.upgrades` in the config). python-sc2's `debug_upgrade` would also grant
  attack and armor upgrades.
- **Without detection, burrowing is a free heal.** A burrowed roach heals 7 life per second and
  the enemy can't touch it. Against scripted roaches (200 fights each) the burrow rule (burrow
  below 40% life, unburrow above 70%) won 98 to 100% of 1 vs 2 and 2 vs 3, and 57 to 68% of
  3 vs 5 with the rest running out of the 160-second limit. Attack and the smart kite rule lost
  every fight.
- **With an Overseer that stays with its army, burrowing is useless.** A burrowed roach moves
  slower than its chasers and can't leave detection; both burrow rules lost every fight (5-fight
  checks).
- **So burrow micro is close to all-or-nothing** in these fights, and a fixed rule already gets
  what there is. Pick units whose rule has clear room to improve before building a task.

What the ravager baselines showed (bile, 2026-09). Bile lands about 1.6 s after the cast (radius
0.5, 60 damage, ready again after about 7.3 s). Rules against scripted roaches, 200 fights each,
1 ravager (win rate):

| scenario | attack | smart (kite, no bile) | bile where it is now | bile with a straight-line lead |
|---|---|---|---|---|
| 1 vs 2, chasing | 0% | 0% | 0% | 4% |
| 1 vs 2, dodging bile | 0% | 0% | 0% | 55% |
| 1 vs 2, leashing | 0% | 100% (23% life lost) | 95% (44% lost) | 98% (42% lost) |
| 1 vs 3, chasing / dodging | 0% | 0% | 0% | 0 to 1% |
| 1 vs 3, leashing | 0% | 78% | 66% | 71% |

- **Bile where the target is now misses anything that moves;** it dealt no more damage than not
  casting at all. A straight-line lead nearly doubled the damage dealt against chasers.
- **Enemies that step out of bile lose more.** While dodging they don't chase or shoot; the lead
  rule won 55% against dodgers but 4% against chasers. That zoning is much of bile's value.
- **No rule is best everywhere.** Against leashing roaches, biling cost life (the ravager steps
  in to cast), so the no-bile kite rule was best. Unlike burrowing, this leaves a model room to
  beat every rule by choosing when and where to bile.
- The scripted dodge needs a reaction time (drawn per fight): a bile is visible about 1.4 s before
  it lands, so an instant dodger escapes every bile. It also has to keep units from walking into
  a bile, not only step out of one.
- **Trained from scratch on 1 vs 2 only, the ravager plateaued.** 2M decisions against chasing,
  dodging and leashing roaches (weights 2:2:1, gamma 0.999). Damage dealt against chasers stayed
  at 41 to 48% from 100k decisions to 2M, and no checkpoint (1M to final, 200 fights each) beat the
  best rule anywhere: chasing 0 to 1% wins (rule 4%), dodging 0 to 6% (rule 55%), leashing 64 to
  81% with 52 to 76% life lost (the no-bile kite rule: 100% with 23%). Likely cause: it almost
  never won the hard fights, so the big reward (life left on a win) never showed up; only the
  small reward for firing often, which doesn't care about surviving. A task needs fights the
  model can win early, like the stalker's 1-roach fights.
- **Winnable 1 vs 1 fights in the mix didn't fix it.** Every rule wins 1 ravager vs 1 roach, but
  life lost differs (attack 86%; against chasers, dodgers, leashers: smart 56/56/10%, the lead rule
  44/22/19%). With these mixed in (weights 1 each next to 1 vs 2 at 2:2:1), the model won 95 to
  100% of its 1 vs 1 training fights from 200k decisions on, but the 500k checkpoint dealt 47% to
  2 chasers and 52% to 2 dodgers (3% wins), no better than before, so the run was stopped there
  as planned. At 600k it lost 62/52/27% of its life in 1 vs 1: kiting a bit worse than the smart
  rule and no sign of aimed bile. Winning alone wasn't enough; the bile skill itself didn't get
  learned. Likely reasons: bile is ready only every 7 s, 8 of 18 actions are biles and most of
  them miss, and a hit lands 12 decisions after the cast, so good aim is rare and hard to credit.
- **Paying for damage and masking bile helped a little, not enough.** Same fights, but the
  reward pays for the share of enemy life taken (bile counts, the reload reward is gone), and the
  8 bile actions are masked while bile isn't ready (sb3-contrib's MaskablePPO). At 500k it dealt
  71% to 2 chasers (earlier runs 47%) and lost 43% of its life against 1 chaser, level with the
  lead rule (44%). Later checkpoints (1M to final, 200 fights each) traded that for the leashers:

  | checkpoint | 1 vs 2 wins: chase / dodge / leash | 1 vs 1 life lost: chase / dodge / leash |
  |---|---|---|
  | 1M | 6 / 12 / 74% | 57 / 46 / 43% |
  | 1.25M | 6 / 6 / 78% | 60 / 53 / 46% |
  | 1.5M | 3 / 2 / 86% | 56 / 62 / 42% (10% lost vs dodgers) |
  | 1.75M | 4 / 0 / 81% | 61 / 54 / 35% |
  | 2M | 8 / 4 / 87% | 53 / 51 / 32% |
  | final | 4 / 6 / 78% | 54 / 49 / 27% |

  Against chasers it is level with the lead rule (4%), not better; against dodgers far below it
  (55%); against leashers below the kite rule (100%), with up to 21% of fights timing out. In
  1 vs 1 it ended up kiting like the smart rule, not biling like the lead rule. So a model can
  learn some bile, but in this mix of fights it doesn't keep it.
- **With the lead rule doing the aiming, the model still fell short of the rules.** In
  RavagerHybridTask both bile actions aim with the straight-line lead; the model only picks when
  to bile, at which of the 2 closest enemies, and how to move and attack (same fights, reward,
  masking and PPO settings as above). Its own lead rule baseline reproduced the rule (50 fights:
  2% against chasers, 50% against dodgers). The checkpoints (1M to final, 200 fights each) won
  0 to 2% against 2 chasers (rule 4%), 4 to 14% against 2 dodgers (rule 55%) and 44 to 69%
  against 2 leashers with 59 to 76% life lost (the kite rule: 100% with 23%). Taking the aiming
  away didn't help: the model didn't learn when a bile pays off.
- **Evaluation noise is large.** 2M and final are 896 decisions apart, almost the same model,
  yet differ by up to 9 points (leashers 87 vs 78%). With 200 fights, treat gaps under about 5
  points near 5% win rates, and under about 10 points near 50%, as chance.

What the zergling baselines showed (group micro, 2026-09). Zerglings with Metabolic Boost against
marines without upgrades that stand and shoot ("chase") or step back while reloading ("kite");
every zergling decides for itself (group task). Rules, 50 to 100 fights each:

| zerglings vs marines | attack-move: wins, life lost (standing / kiting marines) |
|---|---|
| 8v3, 12v5, 16v7 | 100%, 28 to 43% |
| 8v4, 12v6 | 100%, 45 to 50% |
| 8v5 | 100%, 65 / 68% |
| 12v7 | 100%, 66 / 59% |
| 12v8 | 92 / 90%, 85 / 78% |
| 8v6 | 30 / 28% |
| 12v9 | 4 / 12% |

- **The step from winning to losing is one or two marines.** Below about 1.5 zerglings per marine
  attack-move loses most fights; at 2 or more it always wins.
- **Kiting doesn't help marines against speedlings**, which are faster. It barely changes the
  results, as in real games without Stimpack.
- **A flank rule is easy to get wrong.** Running around to the marines' far side before attacking
  matched attack-move against standing marines but lost 83 to 100% against kiting ones: marines
  that keep stepping back are never "passed", so the zerglings circle while getting shot.
- **Trained from scratch, the zerglings never learned to engage.** 8v5, 8v6, 12v8 and 12v9
  against both marine behaviors, one stream per zergling, stopped at 6.85M slot decisions. The
  2.5M checkpoint lost all 800 fights (100 per fight) and dealt 1 to 12% of the marines' life,
  even at 8v5, which attack-move always wins. A melee attack only pays after about 18 decisions in
  a row of running in, and a random start cancels it with a move almost every time, so damage (and
  reward) almost never happened. The budget also looked bigger than it was: slot decisions count
  all 16 slots, so 2.5M were only about 1,000 fights (the ravager saw about 28,000 in 2M).
- **Starting from attack-move fixed the start, not the result.** New models first picking
  "attack closest" 85% of the time and deciding every 9 game loops engaged from the start, but
  the model drifted back to random choices within 180k decisions: empty and dead slots were
  about half of all training examples, and PPO's bonus for varied choices was their only signal.
  With those examples left out, a unit's stream ended at its death, damage credited to the
  zerglings that attacked, and no variety bonus, training wins rose from 12% to 47% by 300k,
  then fell to 0% by 900k (unexplained; fights got longer, as if the zerglings avoided the
  marines). The best checkpoints (250k to 350k, 100 fights each) played like attack-move: 8v5
  and 12v8 same wins and life lost, 8v6 22 to 34% (attack-move 28 to 30%), 12v9 1 to 2% against
  standing marines (4%) and 18 to 25% against kiting ones (12%, from only 50 fights). Not
  counted as a win over the rule.
- **PPO's stability settings stop the collapse; reward scaling makes it worse.** Three settings,
  each run twice (seeds 1 and 2), 2.5M slot decisions. A run "holds" if its training win rate
  over the last 500k is at least 10 points above its start and at most 5 below the 250k to 500k
  window (`tools.rl.stability`):

  | setting | start / 250k to 500k / end, seed 1 | seed 2 | holds |
  |---|---|---|---|
  | learning rate 1e-4 falling to 1e-5, 4 passes, clip 0.1, target_kl 0.02 | 25 / 47 / 56% | 20 / 41 / 54% | both |
  | rewards rescaled while training (VecNormalize) | 28 / 1 / 0% | 25 / 8 / 0% | neither |
  | both | 25 / 38 / 10% | 24 / 43 / 5% | neither |

  The collapsed runs' final models won nothing, and up to half their fights timed out: the
  zerglings stayed away from the marines. The stable models (100 fights per fight) played like
  attack-move: 8v5 97 to 100%, 8v6 23 to 32% (attack-move 28 to 30%), 12v8 80 to 89% (90 to 92%),
  12v9 against standing marines 2 to 4% (4%). Their choices ended up almost fixed, so they most
  likely learned to attack reliably, not to surround. The one repeated difference, 12v9 against
  kiting marines (22 and 26%, and 18 to 25% in the earlier run's best checkpoints, against
  attack-move's 12% from 50 fights), disappeared when measured again with 200 fights each:
  attack-move 16%, the two stable models 20% and 16%.
- **Adding the variety bonus back to the stable setting brought the collapse back.** ent_coef
  0.003 and 0.01, each run twice, 5M slot decisions: all four peaked at 33 to 40% training wins
  (250k to 500k) and none held (end 0 to 26%). At 200 fights on 8v6 and 12v9, three of the final
  models won nothing (the zerglings stayed away from the marines, some fights timing out), and
  the best one lost to attack-move everywhere (8v6 17 / 18% against 28 / 26%, 12v9 4 / 12%
  against 3 / 26%).
- **So far, group RL at our scale reached attack-move and no further.** Training can now be kept
  stable, but only by making the model change slowly and explore little, which leaves it where
  its attack-biased start put it.

The simulator idea, and its headroom check (2026-09). Training in a fast simplified simulator
instead of SC2 would give RL the fights it lacked, but only pays off in a fight where rules leave
room. The check: roaches against marines and marauders without upgrades (a scripted enemy
attacking the closest), every roach deciding for itself (RoachGroupTask), rules that differ only
in their micro. At equal cost attack-move won every fight, so the fights have more bio. 200
fights each:

| rule | 10 roaches vs 12 marines + 4 marauders | 12 vs 14 + 5 |
|---|---|---|
| attack: shoot the closest | 36% | 9% |
| focus: shoot the weakest in range | 38% | 7% |
| threat: shoot the most damage per life left in range | 38% | 12% |
| smart: threat, and pull back below 35% life | 0% | 0% |

- **Target choice changed nothing.** All three attack rules are within noise of each other. In a
  ranged fight on open ground between units this close in range and speed, numbers decide.
- **Pulling back without a heal loses the fight.** A pulled-back roach doesn't heal (no burrow)
  and the chasing bio stays close, so it keeps running instead of shooting (dealt 43 to 46%
  instead of 75 to 88%).
- **So this fight has no headroom for micro**, and a simulator for it isn't worth building.
- **A first run of the focus and threat rules lost everything because of a bug:** unhurt marines
  tie on life, and the enemies come in a different order each step, so the roaches switched
  targets every decision and restarted their attack each time. Target picks now keep the current
  target on ties. Rules that pick targets need this, and so would the bot's.
- **SMAX (JaxMARL) isn't usable as it is:** no roach, no armor or bonus damage (a marauder hits
  a roach for 10 instead of 19), instant attacks, and marines doing 9 damage instead of 6. It has
  its own JAX interface, not Gymnasium, and its speed figures are GPU only. SMAClite models armor
  and bonus damage and has a Gymnasium interface, but isn't faster than SC2. The one transfer
  test found (SMAClite to SMAC) kept roughly half the performance or less.

The headroom survey (2026-09). Four group fights, scripted rules only, to find one where micro
matters and rules are imperfect before any more RL. Bar, fixed beforehand, on sizes where
attack-move wins 15 to 70% (calibrated at 50 fights), 200 fights per rule: (1) the best rule
beats attack-move by 15 points of wins; (2) variants of that rule (one parameter changed)
spread by 15 points. Enemies without upgrades, attacking the closest. Wins, 200 fights each:

| mutalisks (8) vs marines (14) | standing | kiting |
|---|---|---|
| attack | 35% | 51% |
| kite: back off while cooling with a marine within 4.5 | 56% | 52% |
| kite within 6 / 8 | 4 / 0% | 2 / 0% |
| regroup (radius 1, 1.5, 3), then kite within 6 | 3 to 5% | 2% |

| hydralisks (8, speed and range) vs zealots (8) | wins | life lost |
|---|---|---|
| attack | 22% | 96% |
| kite within 2 / 3 / 5 | 70 / 96 / 100% | 85 / 70 / 50% |
| kite within 2 / 3 / 5, shooting the most dangerous | 100% each | 70 / 63 / 45% |

| roaches (6) and ravagers (2) vs 12 marines and 3 marauders | standing in bile | dodging |
|---|---|---|
| attack (no bile) | 7% | 10% |
| bile the closest, lead 1 (closest / most dangerous attack) | 58 / 64% | 54 / 54% |
| bile the most dangerous in range, lead 1 | 52 / 57% | 56 / 56% |
| bile the closest, lead 0.5 | 80 / 82% | 73 / 71% |
| bile the closest, lead 1.5 | 61 / 60% | 51 / 56% |

| burrow roaches vs bio with scans (1 or 2) | 10v16 1 / 2 scans | 12v19 1 / 2 scans |
|---|---|---|
| attack / threat | 30 / 34%, 38 / 30% | 12 / 10%, 14 / 14% |
| burrow below 40%, up above 70% | 14 / 2% | 3 / 0% |
| careful burrow (up when detected) 40/70, 30/60, 50/80 | 57, 56, 49% / 36% each | 20, 20, 16% / 10 to 15% |

- **Mutalisks: kiting only helps at short range, and only against standing marines.** Backing
  off earlier wastes shots against a longer-ranged enemy. Both bars pass as written, but the
  spread comes from worse variants; whether anything beats 4.5 is unmeasured.
- **Hydralisks: rule territory.** Kiting takes attack-move's 22% to 100%; the best rules win
  every fight, so only life lost is left to improve.
- **Burrow with scans: rule territory.** Careful burrowing adds 27 points with one scan and
  nothing with two, and its thresholds barely matter (bar 2 fails). Burrowing without watching
  for scans loses badly. 38 to 81% of careful-burrow fights ran out the 60 s clock.
- **Roaches with ravagers: passes both bars, and the rules aren't at the ceiling.** Bile adds
  45 to 75 points; the lead alone moves wins by 20 points (0.5 best of the three), and whom to
  bile matters too. The best rule wins 82% / 71%, so there is room above it. This is the first
  fight that qualifies as an RL target; the bar for a model is the tuned rule, not these.
- Calibration notes: bio without stim beats roaches with ravagers at equal cost (ravagers are
  poor value without bile), and the 4th marauder flips the fight (10 marines + 3 marauders: 98%,
  + 4: 0%). The scan enemy first never scanned: burrowed roaches drop out of the enemy's unit
  list (the few still listed have type ROACHBURROWED and is_burrowed false), so it now scans
  where a learner unit vanished without dying.

Tuning the bile rule, then training against it (2026-09). Rules first, 200 fights per fight
(standing in bile / dodging):

| bile rule (the most dangerous in range shot otherwise) | standing | dodging |
|---|---|---|
| closest, lead 0 / 0.125 / 0.25 / 0.5 / 0.75 / 1 | 98 / 93 / 81 / 80 / 65 / 56% | 83 / 76 / 72 / 72 / 66 / 59% |
| closest, lead 0, only at a clump of 2 / 3 | 97 / 100% | 84 / 81% |
| the most dangerous in range, lead 0 (the tuned rule) | 96% | 87% |

- **Bile where the target is now.** Any lead does worse; bio mostly stands while it shoots.
  Whom to bile and waiting for clumps change nothing beyond noise. Tuning alone added 11 to 18
  points over the survey's best rule, which is why the bar for a model is the tuned rule.
- **Trained from scratch, the model learned to bile but stayed below the tuned rule.** 3M slot
  decisions (40 minutes), attack-biased start, the zergling "stable" PPO settings, damage
  credited to ravagers while their bile may land. Sampled actions, 200 fights: 2M 78 / 64%,
  final 87 / 74%, against the tuned rule's 96 / 87%. Still rising at the end (+9 / +10 points
  from 2M to 3M) while the learning rate had decayed to its minimum.
- **With 10M slot decisions it reached the tuned rule, not past it.** Same settings, a slower
  learning rate decay (over 10M), 1 h 40 min. Sampled actions, 200 fights: 3M 88 / 68%, 6M
  88 / 71%, 8M 92 / 90%, final 95 / 88%, against the tuned rule's 96 / 87%. Level with the rule
  within noise, far from the +10 the bar asks. The first model to learn a decisive skill from
  scratch and match a tuned rule in a group fight; like every earlier run, it stopped there.
- **With room above the rule, the model still stopped at the rule.** A harder fight, 14 marines
  and 3 marauders (a 4th marauder swung it too far). Re-tuned at 200 fights (standing / dodging):
  attack 0 / 0%, closest lead 0.25 54 / 38%, closest lead 0 80 / 48%, most dangerous lead 0
  80 / 50%, closest lead 0 only at clumps of 2+ 78 / 55% (the bar). The 10M model as it was:
  70 / 54%. Warm-started from it for 5M slot decisions (52 minutes): 2M 78 / 58%, 4M 76 / 54%,
  final 74 / 53%. Level with the rule, falling slightly with more training; the bar asked for
  +10 in one fight. Calibration at 50 fights had given the rule only 56 / 28% here, well below
  its 200-fight result; unexplained, so check sizes at 200 fights before relying on them.
- **Evolution strategies didn't beat the rule either.** A different kind of method: no per-step
  rewards, whole fights scored (win + damage dealt - damage taken), starting from the clump rule
  written as weights (bile the best-scoring seen enemy in range above 0; 8 features: bias,
  closest, clumped, clump size, distance, life, threat, speed). 20 generations of 8 mirrored
  candidates, 30 fights each, 6 SC2 games in parallel: 36 minutes (tools/rl/es.py). The
  starting weights scored 78 / 50% (the rule: 78 / 55%). 200 fights per fight: final weights
  81 / 51%, best generation (10) 85 / 52%. Within noise of the rule; the bar asked for +10. The
  weights barely moved (bias -1.5 to -1.3, small positive weights on distance, life, threat and
  speed), and generation scores swung with the fights' noise: with 30 fights per candidate, the
  search sees little difference between neighbouring bile rules. Near the rule, the fight's
  result doesn't depend much on the bile choice beyond what the rule already does.
- **Evolution strategies found a much better mutalisk rule (the first pass of the bar).** The
  mutalisk fight (8 vs 14 marines, standing / kiting), starting from kite_4.5 written as weights
  (a back-off score and a "shoot the weakest" score over 8 features: bias, cooling, a marine
  within 4.5, cooldown left, closest distance, own life, marines within 6, allies' center
  distance). 20 generations, 8 candidates of 40 fights, 33 minutes. The starting weights scored
  58 / 56% (kite_4.5). The final weights, and the best generation's (13), won 100 / 100% at 200
  fights, losing 66 to 71% of their life, in 10 s fights instead of 14. What it learned: a
  healthy mutalisk only steps back right after shooting and otherwise keeps shooting; a hurt one
  backs off for most of its cooldown (back-off weights: life -0.54, cooldown left +0.58). The
  marines shoot the closest target, so the hurt one stepping back hands their fire to a healthy
  one and spreads the damage. "Shoot the weakest" stayed off. Generation averages swung between
  29 and 99% wins while the best candidate of every generation won nearly everything: the good
  region is narrow, so neighbouring weights play very differently. A hand rule written from
  this insight could likely match it; the search is what found it.
- **A one-line hand rule matches the search, and both carry over.** `life_kite`: back off while
  a marine is within 4.5 and more of the cooldown is left than of the own life (the learned
  boundary, simplified). `hurt_kite_L`: below L of full life kite_4.5, else always shoot. Wins,
  200 fights per fight (standing / kiting marines; the built-in AI, VeryHard, has one fight):

  | 8 mutalisks vs | attack | kite_4.5 | learned | life_kite | hurt_kite_0.5 |
  |---|---|---|---|---|---|
  | 14 marines | 35 / 51% | 58 / 56% | 100 / 100% | 100 / 100% | 100 / 100% |
  | 15 marines | 4 / 6% | 4 / 2% | 90 / 94% | 97 / 100% | 88 / 80% |
  | 16 marines | 0 / 0% | 0 / 0% | 30 / 44% | 38 / 38% | 10 / 14% |
  | 18, 20 marines | 0% | 0% | 0% | 0% | 0% |
  | built-in AI, 14 marines | 38% | 22% | 84% | 84% | 45% |
  | built-in AI, 16 marines | 0% | 0% | 2% | 2% | 0% |

  On the 14-marine fights, hurt_kite_0.7 won 100 / 100% and hurt_kite_0.9 92 / 90%; life_kite
  lost the same share of life as the learned weights (71 / 68%). life_kite matches the learned
  weights everywhere; the plain "only hurt ones kite" rule is weaker away from the tuned fight.
  Against the built-in AI the trick holds (+46 points over attack-move), so it doesn't only
  exploit the scripted enemy's "shoot the closest". The value of the search here was finding
  the idea; the bot would use the rule.
- **Rotation is general against ranged enemies, not against melee ones.** life_kite (distance
  center to center) and the same search (starting from the kite rule at the distance) on two
  earlier fights, 200 fights per fight:

  | roaches vs marines + marauders | 10 vs 12 + 4 | 12 vs 14 + 5 |
  |---|---|---|
  | attack (the best earlier rule) | 36% | 9% |
  | life_kite, enemy within 5 / 7 | 69 / 56% | 30 / 18% |
  | search start (kite within 5) | 21% | 6% |
  | search final / best (16) | 35 / 30% | 11 / 9% |

  | hydralisks vs zealots | wins | life lost |
  |---|---|---|
  | kite within 5, shoot the most dangerous (the best earlier rule) | 100% | 45% |
  | life_kite, within 3 / 5 | 44 / 51% | 93 / 92% |
  | search start (kite within 3) | 98% | 70% |
  | search final / best (20) | 100% / 100% | 47 / 47% |

  - Roaches: the mutalisk rule, unchanged, beats attack-move by 33 and 21 points, in the fight
    the headroom check had called "no room for micro" (that check only tried pulling back below
    35% life until the danger was gone). The search didn't find it: started from kiting, which
    loses here, it drifted to never backing off (attack-move).
  - Hydralisks: rotation hurts against melee (standing to shoot lets zealots connect); the
    search moved toward kiting harder and only matched the best kite rule.
  - So the search finds an idea only when it starts near it; once found, the rule carries over
    to other ranged fights.
- **Rotation follow-ups (remeasured, see the next point): rotation adds to bile in the hard
  fight, the roach rule holds against the built-in AI, and search didn't refine it.** 200
  fights per fight, the configs' own unit order:
  - Bile fight with rotation (life_kite within 5, otherwise the bile rule; standing in bile /
    dodging). The normal fight: most dangerous 98 / 88% became 100 / 92%, clump 100 / 83% became
    99 / 87% (at the ceiling). The hard fight: most dangerous 75 / 52% became 86 / 68%, clump
    75 / 48% became 84 / 60%, +9 to +16. The first gain over the bile rules that PPO and the
    bile search stopped at: they only ever changed the bile decision.
  - The search starting at life_kite_5 (rotate_roach): start 56 / 28%, final 56 / 29%, best
    generation (13) 60 / 26%. No refinement.
  - life_kite_5 measured four times on the roach fight (10v16 / 12v19): 69 / 30, 56 / 28,
    60 / 30 and 64 / 36%. The spread is noise: 200 fights of one rule can land 13 points apart
    near 60%. About 62 / 31% in all, 26 / 22 points above attack-move (36 / 9%).
  - Other sizes (attack / life_kite_5): 10 vs 10 + 4 96 / 98%, 10 vs 14 + 4 0 / 2%, 12 vs
    12 + 5 62 / 82%, 12 vs 16 + 5 0 / 1%. Where there's room (12 vs 17), +20.
  - Built-in AI (each of its 5 builds, 100 fights per fight each, averaged): attack 37 / 12%,
    life_kite_5 55 / 31%, +18 / +19. The roach rule holds against an enemy that doesn't shoot
    the closest target.
- **Two measurement pitfalls, found when one rule scored 14% and 60% against the same enemy.**
  - Unit order in a config sets the formation. All of a side's units spawn at one point and
    SC2 places them outward in the order listed, so `{Marine: 12, Marauder: 4}` puts the
    marauders on the outside (in front) and `{Marauder: 4, Marine: 12}` in the middle. My
    survey script split configs per fight with `yaml.safe_dump`, which sorts keys: every
    per-fight run against mixed bio had the other formation. Against the built-in AI that moved
    life_kite_5 from about 55% to 14% (roaches, 10v16); against the scripted enemy it mattered
    little. Earlier results from those runs (the first rotation follow-ups: bile + rotation
    "8 to 24 points worse", the roach sizes and built-in AI numbers) were wrong and are
    replaced above. The script now keeps the order (`sort_keys=False`).
  - The built-in AI draws a random build per SC2 game, and a game serves hundreds of fights.
    Configs can now set it (`enemy.build`), and `evaluate` / `es` take `--enemy-build`. In these
    fights the build barely matters: every rule scored within a few points across all 5.
- **A search against the built-in AI improved the mutalisk rule.** rotate_mutalisk, starting at
  life_kite, 20 generations against the built-in AI (random builds per worker), 15 minutes.
  Each build separately, 100 fights per fight (14 / 16 marines, averaged over builds): attack
  37 / 0%, kite_4.5 28 / 0%, life_kite 81 / 2%, the search's final weights 93 / 6%: +12 over
  life_kite on average, +9 to +18 per build. Against the scripted enemy it still wins 100 /
  100%. It backs off more (44% of decisions instead of 31%), mostly from a larger weight on
  cooldown left (1.0 to 1.6). The first time a search improved on an already good rule. The
  same search on roaches (rotate_roach, built-in AI) found nothing: 55 / 31% before and after.
- **Against the built-in AI, rotation adds little to the bile rules.** roach_ravager_builtin.yaml
  (no bile dodging), each build, 100 fights per fight, averaged over builds (12 + 3 / 14 + 3
  bio): attack 22 / 0%, most dangerous 94 / 73%, with rotation 97 / 75%; clump 93 / 55%, with
  rotation 96 / 60%. +2 to +5, short of the +10 bar; no build worse. Units rotated on 8% of
  decisions here against 15% against the scripted enemy, where the hard fight gained 9 to 16:
  part of that gain came from the scripted enemy's way of fighting. Bile itself holds up
  (attack-move wins 22 / 0%).
- **Magic box: spacing decides mutalisks against thors, and rotation loses there.** A thor's
  anti-air attack (range 10) splashes, so bunched mutalisks get hit together. Built-in AI
  thors, sizes from calibration (attack-move 26% with 7 mutalisks vs 2 thors, 42% with 11 vs
  3; 6v2 0%, 10v3 10%, 12v3 80%). Each build, 100 fights per fight, averaged (11v3 / 7v2):

  | rule | 11 vs 3 thors | 7 vs 2 thors |
  |---|---|---|
  | attack | 43% | 20% |
  | life_kite | 3% | 1% |
  | spread 1 / 1.5 / 2.5 (while cooling, fly away from an ally that close) | 92 / 93 / 83% | 43 / 39 / 27% |
  | spread 1.5, then life_kite | 91% | 34% |

  Spreading adds 23 to 50 points, and the spacing matters (1 beats 2.5 by 9 and 16), so both
  headroom bars pass. Rotation is useless here: the thors outrange mutalisks (10 against 3), so
  stepping back only loses shots; with spreading it costs 2 to 5. The builds again barely
  differ (within about 10 points of each other per rule).
- **A search starting at spread_1 improved it by 31 points in the harder fight.**
  spread_mutalisk: a spread score (8 features: bias, cooling, an ally within 1, closest ally
  distance, allies within 2, cooldown left, life, closest enemy distance) before the kite
  decisions, starting exactly at spread_1. 20 generations against the built-in AI (random
  builds per worker), 20 minutes. Each build, 100 fights per fight, averaged (11v3 / 7v2):
  start 93 / 44% (spread_1: 92 / 43%), final 100 / 75%, best generation (16) 100 / 74%. Every
  build gained 27 to 35 points in 7v2; life lost fell from 72 to 62% in 11v3. What changed
  (final spread weights): "an ally within 1" rose from 1.0 to 1.5 while "cooling" fell from
  1.0 to 0.5, so a mutalisk now moves off an ally within 1 even when its weapon is ready (the
  count of allies within 2 and the distance to the enemy add to it); back off and the weakest
  stayed off. In words: keep the spacing all the time, not only between shots.
- **A one-line hand rule matches the spread search.** spread_always_s: whenever the closest ally
  is within s, fly straight away from it, weapon ready or not; else attack the closest. Each
  build, 100 fights per fight, averaged (11v3 / 7v2), all in one run:

  | rule | 11 vs 3 thors | 7 vs 2 thors | life lost (11v3) |
  |---|---|---|---|
  | spread_1 (only while cooling) | 94% | 43% | 72% |
  | spread_always_0.75 / 1 / 1.5 | 94 / 100 / 100% | 45 / 75 / 59% | 74 / 63 / 66% |
  | the search's final weights | 100% | 76% | 63% |

  spread_always_1 plays like the learned weights (same wins and life lost against every
  build). The spacing is sharp: 0.75 changes nothing over spread_1, 1.5 gives back 16 points.
  Like life_kite, the search's value was finding the idea; the bot gets a one-line rule.
- **Three more fights through the same gates (calibration, headroom, search), against the
  built-in AI; none passed all of them.** Each build, 100 fights per fight, averaged:
  - **Mutalisks vs marines and thors** (does rotation conflict with spreading?). Calibration:
    attack-move wins 26% with 10 mutalisks vs 6 marines + 2 thors, 46% with 12 vs 8 + 2 (one
    thor: 85 to 100%). Rules (10v6+2 / 12v8+2): attack 25 / 50%, life_kite 5 / 18%,
    spread_always_1 98 / 99%, spread_always_1 then life_kite 98 / 100%. Spreading alone wins
    nearly everything, so no room is left for a search, and the ideas don't conflict here: the
    thors decide the fight.
  - **Zerglings vs marines** (the first RL failure). Calibration: 8 vs 6 marines 35%, 12 vs 9
    12% (5 or 8 marines: 90 to 100%, 7 or 10: none). Rules (8v6 / 12v9): attack 29 / 5%, flank
    27 / 1%, flank radius 2 / 4: 27 / 2% and 16 / 0%. No rule beats attack-move, so the
    headroom gate fails; there is no starting idea to search from.
  - **Bile, searched against the built-in AI** (bile family, starting at the clump rule,
    20 generations, 20 minutes). Final weights 95 / 66% (12 + 3 / 14 + 3 bio): +11 over its
    start (the clump rule, 93 / 55%) in the harder fight, but level with the most dangerous
    rule (94 / 73%, short of the +10 bar). The search climbed to the best hand rule, not past.
- **A survey of four more fights against the built-in AI: one candidate for a search.**
  Calibration in two rounds (the hydralisks won every first-round size). Each build, 100 fights
  per fight, averaged:

  | fight (size) | attack | kite_threat_5 / threat | life_kite_5 | spread_always 1 / 1.5 |
  |---|---|---|---|---|
  | hydralisks vs marines (8 vs 20) | 26% | 57% | 25% | 27 / 0% |
  | hydralisks vs marines + tanks (8 vs 8 + 3) | 78% | 80% | 82% | 82 / 0% |
  | hydralisks vs marines + tanks (8 vs 10 + 3) | 5% | 6% | 4% | 8 / 0% |
  | roaches vs hellbats (8 vs 8) | 38% | 96% | 100% | 37 / 11% |
  | roaches vs marines + tanks (10 vs 8 + 3) | 51% | 55% | 53% | 0 / 0% |

  - **Hydralisks vs marines passes with room:** kiting within 5 while shooting the most
    dangerous adds 31 points (47 to 65% per build), far from 100%. The next search candidate.
  - **Roaches vs hellbats passes but is saturated:** shooting the most dangerous (96%) or
    rotating (100%) wins nearly everything; no room for a search.
  - **Siege tank fights fail the headroom gate:** no rule adds more than 5 points.
  - **Spreading as written doesn't work for ground units:** 1.5 apart loses everything in
    every fight (units keep walking off each other instead of fighting), 1 apart is level or
    worse. Ground units can't overlap (a roach is 1.25 across), so the rule needs a different
    form for them.
- **A search from kite_threat_5 took hydralisks vs 20 marines from 52% to 100%.**
  kite_hydra_threat: the kite family at distance 5 falling back to the most dangerous target,
  starting exactly at kite_threat_5; 20 generations against the built-in AI, 25 minutes. Each
  build, 100 fights: start 52% (the rule measured 57%), final and best (20) 100% against every
  build, life lost 94% to 55%. What it learned: the cooling weight (1.0 to 1.35) now outweighs
  the bias (-1.5 to -1.3) alone, so a hydralisk backs off whenever its weapon cools down, at
  any distance. The starting rule hardly ever kited (0% of decisions): hydralisks shoot from
  about 7 center to center, so a marine was rarely within 5. The search's weights kite on 53%
  of decisions.
- **Any kite limit past the hydralisk's firing distance matches that search.** Each build, 100
  fights against 20 marines, all in one run: kite_threat_5 51% (life lost 94%); kite_threat_7,
  kite_threat_9, kite_threat_any (no limit) and kite_any (no limit, the closest target) 100%
  against every build (life lost 54 to 59%); the search's weights 100% (55%). The whole gain
  was one number: the limit of 5, carried over from the zealot fight, sat below the distance
  hydralisks shoot marines from. Tuning that number by hand would have found it too; the
  search found it without being told which number was wrong.
- **A Zerg survey against Protoss and Zerg: three fights with room for a search.** Built-in AI,
  sizes from up to three calibration rounds (most fights flipped from 100% to 0% between
  neighbouring sizes). Each build, 100 fights per fight, averaged:

  | fight (size) | attack | the other rules |
  |---|---|---|
  | hydralisks vs mutalisks (8 vs 10) | 24% | kite_threat_5 62%, life_kite_5 55%, kite_threat_any 48%, kite_any 5% |
  | roaches vs stalkers (8 vs 7 / 12 vs 10) | 26 / 51% | life_kite_7 79 / 88%, threat 27 / 56%, life_kite_5 27 / 55% |
  | mutalisks vs hydralisks (12 vs 11) | 12% | life_kite 28%, kite_4.5 21%, spread_always_1 0% |
  | roaches vs roaches (8 vs 8) | 61% | life_kite_7 100%, life_kite_5 98%, threat 97% |
  | hydralisks vs stalkers (8 vs 9 / 10 vs 12) | 48 / 21% | kite_threat_5 46 / 21%, life_kite_5 46 / 19%, kite_any 0 / 0% |
  | mutalisks vs phoenixes (12 vs 7) | 63% | life_kite 66%, kite_4.5 12%, spread_always_1 4% |

  - **Pass with room:** hydralisks vs mutalisks (kiting, +38), roaches vs stalkers (rotation at
    7, +34 to +53; at 5 it does nothing: stalkers shoot from 6), mutalisks vs hydralisks
    (rotation, +16, from a low start).
  - **Saturated:** the roach mirror (rotation or threat, 97 to 100%).
  - **No headroom:** hydralisks vs stalkers (equal range: kiting without a limit loses
    everything) and mutalisks vs phoenixes.
  - Rotation (life_kite) helped in 4 of 6 fights, wherever the enemy is ranged and the limit
    covers its range. Spreading lost every non-thor fight.
- **Searches on the three Zerg survey fights: two gains, one loss.** Each from its best rule,
  20 generations against the built-in AI; each build, 100 fights per fight, averaged:
  - **Mutalisks vs hydralisks (12 vs 11), rotate_mutalisk:** start (life_kite) 30%, final 78%
    (+35 to +54 per build), best generation (20) 64%. Learned back-off weights: cooldown left
    1.0 to 1.48, a hydralisk within 4.5 1.5 to 1.37, life -1.0 to -0.86, cooling 0 to 0.33.
    With a hydralisk close, even a healthy mutalisk now backs off in the first half of its
    cooldown (life_kite only let hurt ones), so it kites more (44% of decisions, from 27%).
  - **Hydralisks vs mutalisks (8 vs 10), kite_hydra_threat:** start 61%, final 75%; a repeat
    gave 62% and 73%, so +11 to +15, real but modest. Learned: cooling 1.0 to 1.3, a mutalisk
    within 5 1.0 to 0.65, cooldown left 0 to 0.62: back off early in the cooldown at any
    distance, then come back before the weapon is ready (kite_threat_any, backing off for the
    whole cooldown, measured only 48%).
  - **Roaches vs stalkers (8 vs 7 / 12 vs 10), rotate_roach_7:** start (life_kite_7) 82 / 90%,
    final 31 / 59%. The search walked off the rule: its first generation's candidates (the
    rule with small random changes) won only 24 / 45%, so the rule sits on a narrow peak, and
    following the least-bad candidates shrank the back-off weights until roaches stopped
    rotating (4% of decisions, from 33%). This search can leave a sharp optimum: it judges
    directions by nearby candidates, never by the current weights themselves. The per-build
    check against the start is what catches it.
- **Deterministic evaluation hid all of it:** always taking the most likely action, the same
  models won 6 to 10%, exactly attack-move. At any single step attacking is more likely than
  biling, so the most likely action never biles; sampled, a ready ravager biles within a few
  steps. `evaluate --stochastic` now samples. Earlier group runs (zerglings) were only
  evaluated deterministically, so rare but decisive actions may have been hidden there too.

Tooling pitfalls:

- `debug_show_map` is one game-wide toggle, so only one client may send it; two clients sending it
  switch it off again.
- sharknice's MicroTraining map turns fog off in its map script. Combined with `debug_show_map`,
  enemy units became untargetable snapshots. MicroTraining410 is a 4.10 port without that script.
- SC2 ends every game at game loop 524,288 (2^19, 6.5 game hours); python-sc2 then reports "not
  in a game". One game serves a whole run, so runs past 6.5 game hours per game hit it (once per
  game at 1M decisions). The driver now leaves the game between fights shortly before the limit
  and starts a fresh one. Any game ending with two bots makes one client fail like a crash, so a
  planned restart is marked as such. Unplanned crashes must still end the fight and restart the
  game (raising instead hung the whole training). Stop the two clients of a game together, and
  cancel each only once: python-sc2 exits the process when a request is cancelled twice.
- A fight cut off by the time limit must end with a real observation (the last one), not a blank
  one: PPO estimates the rest of the fight from it, and a blank one made that estimate arbitrary.
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
