# VGC Oracle

> A reinforcement learning agent that plays competitive VGC doubles (Gen 9, Reg I) on a local Pokémon Showdown server.

## Status
Training works and results are below. They come from the training battles.

## How it works
- The agent trains against three scripted bots from poke-env: `RandomPlayer`, `MaxBasePowerPlayer` and `SimpleHeuristicsPlayer`, one worker each.
- Every turn the battle is encoded into numbers: 12 Pokémon slots (own and opponent), 4 moves per Pokémon and field state like weather and field conditions.
- A PyTorch network trained with MaskablePPO (sb3-contrib) picks one action per active Pokémon. Illegal actions are masked out, and pairs of actions that are only illegal together (both Pokémon switching to the same teammate, both terastallizing) are fixed in `wrapper.py`.
- Teams come from a pool of 582 Regulation I teams, so the agent can't just learn one matchup.

## Results
| Opponent               | Win rate      |
|------------------------|---------------|
| RandomPlayer           | 91.4% ± 2.5%  |
| MaxBasePowerPlayer     | 53.4% ± 4.4%  |
| SimpleHeuristicsPlayer | 43.8% ± 4.3%  |

Win rate over the last 500 training battles per opponent at ~23.5M steps. ± is a 95% confidence interval.
The agent was still sampling actions during these battles, so these are training numbers, not best case.

Beating `RandomPlayer` mostly shows the agent works. `SimpleHeuristicsPlayer` is the meaningful result and the agent still loses slightly more than it wins against it.

## Limitations
- The agent is tested against the same three bots it trains against, and on the same team pool.
- Scripted bots are not human players. These numbers say nothing about ladder play.
- Reward weights in `src/config.py` were picked by hand, not tuned.
- The two active Pokémon pick their actions independently, so the agent can't coordinate them well.
- Windows only (`run.py` uses `taskkill`).

## Setup
Needs Python 3.12 and Node.js.

1. Clone the Showdown server anywhere and check out the version used for this project:
   ```
   git clone https://github.com/smogon/pokemon-showdown.git
   cd pokemon-showdown
   git checkout 00c061991c64240ffc939f1e450316182f8764f6
   npm install
   cp config/config-example.js config/config.js
   ```
2. Set `SHOWDOWN_DIR` in `src/config.py` to the full path of that folder.
3. Install Python packages (GPU users: install torch first, see `requirements.txt`):
   ```
   pip install -r requirements.txt
   ```

## Training
Settings are in `src/config.py`. Before a new run, set:
- `RUN_NAME`: the model and logs are saved to `src/model/<RUN_NAME>/` and `src/logs/<RUN_NAME>/`.
  A new name starts from scratch, and an existing name continues that run.
- `TARGET_TIMESTEPS`: training stops at this many steps. `None` trains forever.

Then run:
```
python run.py
```
This starts the Showdown server and runs `src/train.py` in slices of `TOTAL_TIMESTEPS` steps, restarting the server every 8 slices, until the target is reached.

Battle results go to `src/logs/<RUN_NAME>/battles.csv`. Split win rates by opponent instead of using the overall average.

## Project layout
```
run.py              runs training in slices and restarts the server
src/
    config.py       showdown path, reward weights and training settings
    train.py        MaskablePPO training
    env.py          gym environment: observation, action space, reward
    wrapper.py      splits out the action mask, fixes jointly illegal actions
    extractor.py    network that turns the observation into features
    teambuilder.py  team pool
    metrics.py      per-battle logging and stall detection
    encode/         battle state -> numbers (pokemon, moves, battle, state)
    teams/          582 team pastes
```

## Tech
Python · PyTorch · Stable-Baselines3 / sb3-contrib · poke-env · Pokémon Showdown
