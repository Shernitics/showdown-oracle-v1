# VGC Oracle

> A reinforced-learning agent that plays competitive VGC doubles (Gen 9, Reg I) on a local Pokémon Showdown server.

## Status
Complete with minimal polishing left.

## How it works
- The agent trains against three scripted bots from poke-env: `RandomPlayer`, `MaxBasePowerPlayer` and `SimpleHeuristicPlayer`, one worker each.
- Every turn the battle is encoded into numbers: 12 Pokémon slots (own and opponent), 4 moves per Pokémon and field state like weather and field conditions.
- A PyTorch network trained with MaskablePPO (sb3-contrib) picks one action per active Pokémon. Illegal actions are masked out.
- Teams come from a pool of 582 Regulation I teams.

## Results
| Opponent               | Win rate |
|------------------------|----------|
| RandomPlayer           | TBD      |
| MaxBasePowerPlayer     | TBD      |
| SimpleHeuristicsPlayer | TBD      |

# Setup
Windows only (`run.py` uses `taskkill`). Needs Python 3.12 and Node.js.

1. Clone the Showdown server into your Desktop folder (`run.py` expects `)
   ```
   git clone https://github.com/smogon/pokemon-showdown.git
   cd pokemon-showdown
   git checkout <commit>
   npm install
   cp config/config-example.js config/config.js
   ```
2. Set `SHOWDOWN_DIR` in `src/config.py` to the full path of that folder.
3. Install Python packages (GPU users: install torch first, see `requirements.txt`)
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
This starts the Showdown server and runs `src/train.py` and in slices of `TOTAL_TIMESTEPS` steps, restarting the server every 8 slices, until the target is reached.

battle results got to `src/logs/<RUN_NAME>/battles.csv`. Split win rates by opponent instead of using the overall average.

## Project layout
```
run.py              runs training in slices and restarts server
src/
    config.py       showdown path, reward weights and training settings
    train.py        MaskablePPO training
    env.py          gym environment: observation, action space, rward
    wrapper.py      splits out the aciton mask, fixes jointly illegal actions
    extractor.py    network that turns the observation into features
    teambuilder.py  team pool
    metrics.py      per-battle logging and stall detection
    encode/         battle state -> numbers (pokemon, moves, battle. state)
    teams/          582 team pastes
```

## Tech
Python · PyTorch · Stable-Baselines3 / sb3-contrib · poke-env · Pokémon Showdown
