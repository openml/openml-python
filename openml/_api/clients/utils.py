from __future__ import annotations

import math
import random


def robot_delay(n: int) -> float:
    """
    Compute delay for automated retry policy.

    Parameters
    ----------
    n : int
        Current retry attempt number (1-based).

    Returns
    -------
    float
        Number of seconds to wait before the next retry.
    """
    wait = (1 / (1 + math.exp(-(n * 0.5 - 4)))) * 60
    variation = random.gauss(0, wait / 10)
    return max(1.0, wait + variation)


def human_delay(n: int) -> float:
    """
    Compute delay for human-like retry policy.

    Parameters
    ----------
    n : int
        Current retry attempt number (1-based).

    Returns
    -------
    float
        Number of seconds to wait before the next retry.
    """
    return max(1.0, n)
