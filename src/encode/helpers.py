"""
Small shared utilities for the encode modules.
"""


def normalize(value:int | float, cap: int | float) -> int | float:
    """
    Scale a value into [0, 1] by dividing its cap.
    The value is clamped first, so anything below 0 becomes 0.0 and anything above the cap becomes 1.0.

    :param value: raw count, duration or stat.
    :param cap: the larget meaningful value (usually from vocab.py). Must be > 0.
    :return: float in [0, 1]
    """
    return min(max(float(value), 0.0), cap) / cap