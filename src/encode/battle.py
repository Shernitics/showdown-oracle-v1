"""
Encodes the battle state. Everything that belongs to the field or the side rather than to a single Pokémon (turn, weather, field, side conditions, tera).
"""


import numpy as np
from poke_env.battle import DoubleBattle

from encode.vocab import *
from encode.helpers import normalize

ENVIRONMENT_FEATURES_CONT = 82
BROUGHT_SIZE = 4

def encode_battle(battle: DoubleBattle):
    """
    Encode the field and side state of a doubles battle as a vector.
    Every feature is in [0, 1]: flags are 0/1, and counts and durations are normalized by caps in vocab.py.
    Durations are 'turns elapsed since it started / max length'. Poke-env stores the turn each condition started, not the turns left.

    Layout (82 floats, in order):
        [0:4]       battle: turn / 100, in team preview, own Pokémon left / 4, opponent Pokémon left / 4
        [4:13]      weather: 8 flags. then the active weather's duration
        [13:29]     field: 8 flags, then 8 durations
        [29:69]     side conditions, own then opponent. Each side has 13 flags then 7 durations
        [69:73]     stackable side conditions, own then opponent
        [73:82]     tera and turn status: can_tera x2, used_tera (own, opp), force_switch x2, trapped x2, reviving (Revival Blessing)

    :param battle: current poke-env doubles battle, seen from our side.
    :return: {"cont": float32 array of shape (ENVIRONMENT_FEATURES_CONT,)}, env.py exposes it as the "battle_cont" observation.
    """

    turn = battle.turn

    team_preview = battle.teampreview
    turn_norm = normalize(turn, 100)
    pokemon_remaining = normalize(BROUGHT_SIZE - sum(1 for p in battle.team.values() if p.fainted), BROUGHT_SIZE)
    pokemon_remaining_opp = normalize(BROUGHT_SIZE - sum(1 for p in battle.opponent_team.values() if p.fainted), BROUGHT_SIZE)

    # weather
    weather = [float(w in battle.weather) for w in WEATHER_DURATION_CAPS.keys()]
    current_weather = list(battle.weather)[0] if battle.weather else None
    weather_duration_max = WEATHER_DURATION_CAPS.get(current_weather)
    weather_duration = normalize(turn - battle.weather[current_weather], weather_duration_max) if weather_duration_max else 0.0

    # field
    field = [float(f in battle.fields) for f in FIELD_DURATION_CAPS.keys()]
    field_duration = [normalize(turn - battle.fields.get(f, turn), c) for f, c in FIELD_DURATION_CAPS.items()]

    # side conditions
    side_conditions = [float(s in battle.side_conditions) for s in SIDE_CONDITION_DURATION_CAPS.keys()]
    side_conditions_duration = [normalize(turn - battle.side_conditions.get(s, turn), c) for s, c in SIDE_CONDITION_DURATION_CAPS.items() if c is not None]
    side_conditions_opp = [float(s in battle.opponent_side_conditions) for s in SIDE_CONDITION_DURATION_CAPS.keys()]
    side_conditions_duration_opp = [normalize(turn - battle.opponent_side_conditions.get(s, turn), c) for s, c in SIDE_CONDITION_DURATION_CAPS.items() if c is not None]

    # side conditions (stackable)
    side_conditions_stackable = [normalize(battle.side_conditions.get(s, 0), c) for s, c in SIDE_CONDITION_STACKABLE_CAPS.items()]
    side_conditions_stackable_opp = [normalize(battle.opponent_side_conditions.get(s, 0), c) for s, c in SIDE_CONDITION_STACKABLE_CAPS.items()]

    # misc
    can_tera = battle.can_tera
    used_tera = [battle.used_tera, battle.opponent_used_tera]
    force_switch = battle.force_switch
    trapped = battle.trapped
    reviving = battle.reviving

    cont = np.asarray(
        [

            # battle
            turn_norm, team_preview, pokemon_remaining, pokemon_remaining_opp,

            # weather
            *weather, weather_duration,

            # field
            *field, *field_duration,

            # side conditions
            *side_conditions, *side_conditions_duration, *side_conditions_opp, *side_conditions_duration_opp,

            # side conditions (stackable)
            *side_conditions_stackable, *side_conditions_stackable_opp,

            # misc
            *can_tera, *used_tera, *force_switch, *trapped, reviving,

        ]
    , dtype=np.float32)

    return {"cont": cont}
