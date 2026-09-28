"""
Encodes a single move. Its power, typing, targeting, side effects and remaining PP.
Each Pokémon has 4 move slots; encode/state.py calls this for every slot on all 12 Pokémon.
"""


import numpy as np
from poke_env.battle import Move

from encode.vocab import *
from encode.helpers import normalize

MOVE_FEATURES_CONT = 92
MOVE_FEATURES_CAT = 1

def encode_move(move: Move):
    """
    Encode one move as a continuous feature plus a move ID for the embedding layer.
    Every continuous feature is in [0, 1].

    An empty slot (move is None, e.g. unrevealed opponent moves) returns all zeros.
    Index 0 is a presence flag (always 1.0 for a real move), so the network can tell "no move" apart from a real move whose features are mostly 0.

    Layout (92 floats, in order):
        [0]         presence flag
        [1:5]       base power / 250, accuracy, priority, PP left / max PP
        [5:8]       category one-hot (physical, special, status)
        [8:27]      type one-hot (poke-env Target order)
        [27:42]     target one-hot (poke-env Target order)
        [42:59]     flags
        [59:66]     stat changes applied to the target
        [66:73]     stat changes applied to the user
        [73:81]     status inflicted one-hot
        [81:86]     expected hits / 5, crit ratio / 6, drain, recoil, heal (fractions of HP)
        [86:92]     force switch, user switch, breaks protect, is protect, ignores ability, thaws target

    :param move: a poke-env Move, or None for an empty encoding.
    :return:    {"cont": float32 array of shape (MOVE_FEATURES_CONT,), "cat": {"name": int64 array of shape (1,)}}.
                "name" is the MOVE_NUM id and 0 means empty slot or unknown move.
    """


    if move is None:
        return {
            "cont": np.zeros(MOVE_FEATURES_CONT, dtype=np.float32),
            "cat": {
                "name": np.zeros(1, dtype=np.int64),
            }
        }

    cat = {
        "name": np.asarray([MOVE_NUM.get(move.id, 0)], dtype=np.int64),
    }


    # move characteristics
    base_power = normalize(move.base_power, 250)
    accuracy = move.accuracy
    priority = normalize(move.priority + 7, 12)
    max_pp = move.max_pp
    pp = normalize(move.current_pp, max_pp) if max_pp else 0.0

    # poke-env rebuilds these on every access, so read them once
    move_category = move.category
    type_name = move.type.name.capitalize()
    move_target = move.target
    move_flags = move.flags

    category = [float(move_category is c) for c in MOVE_CATEGORIES]
    type = [float(t == type_name) for t in TYPES]
    target = [float(move_target is t) for t in MOVE_TARGETS]
    flags = [float(f in move_flags) for f in MOVE_FLAGS]

    # stat changes
    move_boosts = move.boosts or {}
    move_self_boost = move.self_boost or {}
    boosts = [normalize(move_boosts.get(k, 0) + 6, 12) for k in BOOST_KEYS]
    self_boost = [normalize(move_self_boost.get(k, 0) + 6, 12) for k in BOOST_KEYS]

    # status
    move_status = move.status
    current_status = move_status.name if move_status is not None else None
    status = [float(n == current_status) for n in STATUS_DURATION_CAPS]

    # damage profile
    expected_hits = normalize(move.expected_hits, 5)
    crit_ratio = normalize(move.crit_ratio, 6)
    drain = move.drain
    recoil = move.recoil
    heal = move.heal

    # behavior
    force_switch = float(move.force_switch)
    self_switch = float(bool(move.self_switch))
    breaks_protect = float(move.breaks_protect)
    is_protect_move = float(move.is_protect_move)
    ignore_ability = float(move.ignore_ability)
    thaws_target = float(move.thaws_target)


    cont = np.asarray(
        [
            1.0,

            # move characteristics
            base_power, accuracy, priority, pp, *category, *type, *target, *flags,

            # stat changes
            *boosts, *self_boost,

            # status
            *status,

            # damage profile
            expected_hits, crit_ratio, drain, recoil, heal,

            # behaviour
            force_switch, self_switch, breaks_protect, is_protect_move, ignore_ability, thaws_target,

        ], dtype=np.float32
    )

    return {
        "cont": cont,
        "cat": cat
    }
