"""dm.py

Difference-in-means (DM) baseline statistics.

Implements the test statistic defined in the baselines:
for (t,l), outcome at u=t+l, treated group {A=t}, controls {A>u}.

This module contains only statistic computations (no nuisance fitting).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from helpers import (
    SADDesign,
    dm_stat_global_from_A,
    dm_stats_global_from_perms,
    dm_stat_tl_from_A,
    dm_stats_tl_from_perms,
)

def observed_global(
    A: np.ndarray,
    Y: np.ndarray,
    design: SADDesign,
    subset: Optional[np.ndarray]=None,
) -> float:
    return dm_stat_global_from_A(A, Y, design.T, subset=subset)

def perms_global(
    A_perms: np.ndarray,
    Y: np.ndarray,
    design: SADDesign,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    return dm_stats_global_from_perms(A_perms, Y, design.T, subset=subset)

def observed_tl(
    A: np.ndarray,
    Y: np.ndarray,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    return dm_stat_tl_from_A(A, Y, t=t, l=l, subset=subset)

def perms_tl(
    A_perms: np.ndarray,
    Y: np.ndarray,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    return dm_stats_tl_from_perms(A_perms, Y, t=t, l=l, subset=subset)
