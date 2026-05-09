"""cdm.py

Covariate-adjusted difference-in-means (cDM) baseline.

We estimate mu_t(x)=E[Y_t|X=x] by regression (no A used), then residualize
outcomes: Y_t^adj = Y_t - mu_hat_t(X). DM statistics are then computed using
Y^adj in place of Y.

This preserves the "no A in nuisance" property.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from helpers import (
    SADDesign,
    dm_stat_global_from_A,
    dm_stat_tl_from_A,
    dm_stats_global_from_perms,
    dm_stats_tl_from_perms,
    fit_mu_models,
    predict_mu,
    wrap_fixed_mu_models,
)

@dataclass
class CDMResult:
    mu_models: List
    Y_adj: np.ndarray

def fit_cdm(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mu_degree: int=1,
    ridge_alpha: float=0.001,
    option: str='parametric',
    assumption: Optional[str]=None,
    random_state: Optional[int]=12345,
    mu_gbrt_params: Optional[Dict[str, object]]=None,
    true_mu: Optional[np.ndarray]=None,
) -> CDMResult:
    if true_mu is not None:
        mu_hat = np.asarray(true_mu, dtype=float)
        mu_models = wrap_fixed_mu_models(X, mu_hat)
    else:
        mu_models = fit_mu_models(
            X,
            Y,
            degree=mu_degree,
            ridge_alpha=ridge_alpha,
            option=option,
            assumption=assumption,
            random_state=random_state,
            gbrt_params=mu_gbrt_params,
        )
        mu_hat = predict_mu(mu_models, X)
    Y_adj = Y - mu_hat
    return CDMResult(mu_models=mu_models, Y_adj=Y_adj)

def observed_global(
    A: np.ndarray,
    Y_adj: np.ndarray,
    design: SADDesign,
    subset: Optional[np.ndarray]=None,
) -> float:
    return dm_stat_global_from_A(A, Y_adj, design.T, subset=subset)

def perms_global(
    A_perms: np.ndarray,
    Y_adj: np.ndarray,
    design: SADDesign,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    return dm_stats_global_from_perms(A_perms, Y_adj, design.T, subset=subset)

def observed_tl(
    A: np.ndarray,
    Y_adj: np.ndarray,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    return dm_stat_tl_from_A(A, Y_adj, t=t, l=l, subset=subset)

def perms_tl(
    A_perms: np.ndarray,
    Y_adj: np.ndarray,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    return dm_stats_tl_from_perms(A_perms, Y_adj, t=t, l=l, subset=subset)
