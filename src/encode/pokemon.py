"""
Encode a Pokémon. Its species data, current stats and HP, tera, status and volatile effects.
"""


import numpy as np
from poke_env.battle import Pokemon

from encode.vocab import *
from encode.helpers import normalize

# cont - continuous features (float32); cat - categorical features (int64)
POKEMON_FEATURES_CONT = 110
POKEMON_FEATURES_CAT = 3


def encode_pokemon(pokemon: Pokemon | None, position: int | None):
    """
    Encode one Pokémon as a continuous feature plus three IDs for abilities, items and species for embedding layers.
    Every continuous feature is in [0, 1].

    An empty slot (Pokémon is None) returns all zeros. Index 0 is a presence flag, so a network can tell an empty slot apart from a real Pokémon.

    Layout (110 floats, in order):
        [0]         presence flag
        [1:4]       position: left slot, right slot, not on the field
        [4:10]      base stats / 255 (hp, atk, def, spa, spd, spe)
        [10:29]     type flags (TYPES order)
        [29]        current HP fraction
        [30]        stats known flag
        [31:37]     actual stats / REAL_STAT_CAP (0.0 if unknown)
        [37:44]     stat boosts (BOOST_KEYS order)
        [44:52]     level / 100, gender x3, must recharge, first turn on field, revealed, brought
        [52:71]     tera type one-hot (TYPES order)
        [71]        terastallized flag
        [72:80]     status one-hot (STATUS_DURATION_CAPS order, NONE included)
        [80]        status duration (sleep / toxic counter)
        [81:101]    volatile effect flags (EFFECT_DURATION_CAPS order)
        [101:105]   effect durations (taunt, encore, disable, magnet rise)
        [105:109]   perish song count (3, 2, 1, 0)
        [109]       protect counter (consecutive protects)

    :param pokemon: a poke-env Pokémon, or None for an empty encoding.
    :param position: 0 = left active slot, 1 = right active slot, None = not on the field
    :return:    {"cont": float32 array of shape (POKEMON_FEATURES_CONT),
                "cat":  {{"species": int64 array of shape (1,)},
                        {"items": int64 array of shape (1,)},
                        {"ability": int64 array of shape (1,)}}
                IDs are 0 for an empty slot or an unknown value. Items also have 1 = held but not yet revealed, and 2 = holds no item.
    """


    if pokemon is None:
        return {
            "cont": np.zeros(POKEMON_FEATURES_CONT, dtype=np.float32),
            "cat": {
                "species": np.zeros(1, dtype=np.int64),
                "items": np.zeros(1, dtype=np.int64),
                "ability": np.zeros(1, dtype=np.int64),
            }
        }

    cat = {
        "species": np.asarray([SPECIES_NUM.get(pokemon.species, 0)], dtype=np.int64),
        "items": np.asarray([ITEM_NUM.get(pokemon.item, 0)], dtype=np.int64),
        "ability": np.asarray([ABILITY_NUM.get(pokemon.ability, 0)], dtype=np.int64),
    }

    # position: left, right, unknown
    active = [float(position == 0), float(position == 1), float(position is None)]

    # pokemon characteristics (universal for this species)
    base_stats = [normalize(s, BASE_STAT_CAP) for s in pokemon.base_stats.values()]
    pokemon_types = {t.name.capitalize() for t in pokemon.types}
    type = [float(type_name in pokemon_types) for type_name in TYPES]

    # stats
    is_revealed = pokemon.revealed
    pokemon_stats = pokemon.stats
    pokemon_boosts = pokemon.boosts
    current_hp_fraction = pokemon.current_hp_fraction if is_revealed else 1.0
    stats_known = 1.0 if pokemon_stats.get("atk") is not None else 0.0
    stats = [normalize(v, REAL_STAT_CAP.get(s)) if stats_known else 0.0 for s, v in pokemon_stats.items()]
    boosts = [normalize(pokemon_boosts[k] + 6, 12) for k in BOOST_KEYS]

    # pokemon oriented
    pokemon_gender = pokemon.gender
    level = normalize(pokemon.level, 100)
    gender = [1.0 if pokemon_gender is g else 0.0 for g in GENDERS]
    must_recharge = float(pokemon.must_recharge)
    first_turn = float(pokemon.first_turn)
    revealed = float(is_revealed)
    brought = float(pokemon.selected_in_teampreview or is_revealed)

    # terastallization
    tera = pokemon.tera_type
    tera_name = tera.name.capitalize() if tera is not None else None
    tera_type = [float(t == tera_name) for t in TYPES]
    is_terastallized = float(pokemon.is_terastallized)

    # status / effects
    pokemon_status = pokemon.status
    current_status = pokemon_status.name if pokemon_status else "NONE"
    status_name = [float(n == current_status) for n in STATUS_DURATION_CAPS]
    status_duration_max = STATUS_DURATION_CAPS.get(current_status)
    status_duration = normalize(pokemon.status_counter, status_duration_max) if status_duration_max else 0.0

    pokemon_effects = pokemon.effects
    current_effects = [float(e in pokemon_effects) for e in EFFECT_DURATION_CAPS]
    effects_duration = [normalize(pokemon_effects.get(e, 0), c) for e, c in EFFECT_DURATION_CAPS.items() if c is not None]
    perish_effects = [(p in pokemon_effects) for p in PERISH_EFFECTS]
    protect_counter = normalize(pokemon.protect_counter, PROTECT_COUNTER_CAP)


    cont = np.asarray(
        [
            1.0, *active,

            # pokemon characteristics (universal for this species)
            *base_stats, *type,

            # combat relevance

            # stats
            current_hp_fraction, stats_known, *stats, *boosts,

            # pokemon oriented
            level, *gender, must_recharge, first_turn, revealed, brought,

            # terastallization
            *tera_type, is_terastallized,

            # status / volatile
            *status_name, status_duration, *current_effects, *effects_duration, *perish_effects, protect_counter

        ]
    , dtype=np.float32)

    return {
        "cont": cont,
        "cat": cat
    }
