"""Sample-splitting baseline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from helpers import (
    LikelihoodScoreComponents,
    SADDesign,
    build_likelihood_score_components,
    fit_mu_models,
    likelihood_score_stat_global_from_A,
    likelihood_score_stat_tl_from_A,
    likelihood_score_stats_global_from_perms,
    likelihood_score_stats_tl_from_perms,
    make_regressor,
    poly_basis_transformer,
    predict_mu,
    wrap_fixed_mu_models,
)

@dataclass
class SSSplit:
    est_idx: np.ndarray
    test_idx: np.ndarray
    mu_models: List
    comps: LikelihoodScoreComponents

@dataclass
class SSResult:
    splits: List[SSSplit]

    @property
    def folds(self) -> List[SSSplit]:
        return self.splits

def _fit_tau_ss_linear(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    design: SADDesign,
    est_idx: np.ndarray,
    mu_models: List,
    *,
    basis_degree: int=1,
    ridge_alpha: float=0.001,
    option: str='parametric',
    random_state: Optional[int]=0,
    true_mhat: Optional[Dict[Tuple[int, int], np.ndarray]]=None,
) -> np.ndarray:
    T, pi, pi_leq = (design.T, design.pi, design.pi_leq)
    R_est = Y[est_idx] - predict_mu(mu_models, X[est_idx])
    A_est = A[est_idx]
    mhat: Dict[Tuple[int, int], np.ndarray] = {}
    if true_mhat is not None:
        mhat = {k: np.asarray(v, dtype=float).copy() for k, v in true_mhat.items()}
    elif option == 'parametric':
        from sklearn.linear_model import Ridge
        Phi = poly_basis_transformer(degree=basis_degree).fit_transform(X)
        Phi_est = Phi[est_idx]
        reg = Ridge(alpha=ridge_alpha, fit_intercept=False)
        for t in range(1, T + 1):
            Rt = R_est[:, t - 1]
            for k in range(t):
                a = t - k
                if pi[a] <= 0:
                    continue
                reg.fit(Phi_est, Rt * ((A_est == a).astype(float) - pi[a]))
                mhat[t, k] = Phi @ reg.coef_.reshape(-1)
    else:
        X_est = X[est_idx]
        rs0 = None if random_state is None else int(random_state)
        for t in range(1, T + 1):
            Rt = R_est[:, t - 1]
            for k in range(t):
                a = t - k
                if pi[a] <= 0:
                    continue
                reg = make_regressor(
                    degree=basis_degree,
                    ridge_alpha=ridge_alpha,
                    option=option,
                    random_state=None if rs0 is None else rs0 + 1000 * t + k,
                )
                reg.fit(X_est, Rt * ((A_est == a).astype(float) - pi[a]))
                mhat[t, k] = reg.predict(X)
    tau_hat = np.zeros((X.shape[0], T), dtype=float)
    for k in range(T):
        vals = []
        for t in range(k + 1, T + 1):
            a, denom = (t - k, 1.0 - pi_leq[t])
            if pi[a] <= 0 or denom <= 0 or (t, k) not in mhat:
                continue
            bias = sum((mhat[key] for key in ((t, j) for j in range(t)) if key in mhat)) / denom
            vals.append(mhat[t, k] / pi[a] + bias)
        tau_hat[:, k] = 0.0 if not vals else np.mean(np.stack(vals), axis=0)
    return tau_hat if np.nanmean(tau_hat[:, 0]) >= 0 else -tau_hat

def fit_ss_split(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    design: SADDesign,
    *,
    mu_degree: int=1,
    basis_degree: int=1,
    mu_ridge_alpha: float=0.001,
    tau_ridge_alpha: float=0.001,
    split_seed: Optional[int]=None,
    option: str='parametric',
    assumption: Optional[str]=None,
    true_mu: Optional[np.ndarray]=None,
    true_mhat: Optional[Dict[Tuple[int, int], np.ndarray]]=None,
) -> SSResult:
    rng = np.random.default_rng(split_seed)
    est_idx, test_idx = np.split(rng.permutation(X.shape[0]), [X.shape[0] // 2])
    if true_mu is not None:
        mu_models = wrap_fixed_mu_models(X, np.asarray(true_mu, dtype=float))
    else:
        mu_models = fit_mu_models(
            X,
            Y,
            degree=mu_degree,
            ridge_alpha=mu_ridge_alpha,
            option=option,
            assumption=assumption,
        )
    tau_hat = _fit_tau_ss_linear(
        X,
        Y,
        A,
        design,
        est_idx=est_idx,
        mu_models=mu_models,
        basis_degree=basis_degree,
        ridge_alpha=tau_ridge_alpha,
        option=option,
        random_state=split_seed,
        true_mhat=true_mhat,
    )
    comps = build_likelihood_score_components(design, X, Y, mu_models, tau_hat)
    return SSResult([SSSplit(est_idx=est_idx, test_idx=test_idx, mu_models=mu_models, comps=comps)])

def _permute_within_sets(
    A_obs: np.ndarray,
    sets: List[np.ndarray],
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    P = np.empty((n_perms, A_obs.shape[0]), dtype=int)
    for r in range(n_perms):
        out = A_obs.copy()
        for idx in sets:
            out[idx] = rng.permutation(out[idx])
        P[r] = out
    return P

def _split_sets(
    res: SSResult,
    A_obs: np.ndarray,
    *,
    t: Optional[int]=None,
    l: Optional[int]=None,
    T: Optional[int]=None,
    subgroup_idx: Optional[np.ndarray]=None,
) -> List[np.ndarray]:
    sets: List[np.ndarray] = []
    for split in res.splits:
        idx = split.test_idx if subgroup_idx is None else np.intersect1d(split.test_idx, subgroup_idx, assume_unique=False)
        if t is not None and l is not None and (T is not None):
            u = t + l
            idx = idx[(A_obs[idx] == t) | (A_obs[idx] > u)]
        sets.append(idx)
    return sets

def perms_H0_ss(
    res: SSResult,
    A_obs: np.ndarray,
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    return _permute_within_sets(A_obs, _split_sets(res, A_obs), rng, n_perms)

def perms_H_tl_ss(
    res: SSResult,
    A_obs: np.ndarray,
    t: int,
    l: int,
    T: int,
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    return _permute_within_sets(A_obs, _split_sets(res, A_obs, t=t, l=l, T=T), rng, n_perms)

def perms_H_0k_ss(
    res: SSResult,
    A_obs: np.ndarray,
    subgroup_idx: np.ndarray,
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    return _permute_within_sets(A_obs, _split_sets(res, A_obs, subgroup_idx=subgroup_idx), rng, n_perms)

def _apply_score(
    res: SSResult,
    obs_fn,
    perm_fn,
    A_like: np.ndarray,
    subgroup_idx: Optional[np.ndarray]=None,
    **kwargs,
):
    vals = []
    for split in res.splits:
        subset = split.test_idx if subgroup_idx is None else np.intersect1d(split.test_idx, subgroup_idx, assume_unique=False)
        vals.append((perm_fn if A_like.ndim == 2 else obs_fn)(A_like, split.comps, subset=subset, **kwargs))
    arr = np.vstack(vals) if A_like.ndim == 2 else np.asarray(vals, dtype=float)
    return arr.mean(axis=0) if A_like.ndim == 2 else float(arr.mean())

def observed_global(res: SSResult, A: np.ndarray) -> float:
    return _apply_score(res, likelihood_score_stat_global_from_A, likelihood_score_stats_global_from_perms, A)

def perms_global(res: SSResult, A_perms: np.ndarray) -> np.ndarray:
    return _apply_score(res, likelihood_score_stat_global_from_A, likelihood_score_stats_global_from_perms, A_perms)

def observed_tl(res: SSResult, A: np.ndarray, t: int, l: int) -> float:
    return _apply_score(res, likelihood_score_stat_tl_from_A, likelihood_score_stats_tl_from_perms, A, t=t, l=l)

def perms_tl(res: SSResult, A_perms: np.ndarray, t: int, l: int) -> np.ndarray:
    return _apply_score(res, likelihood_score_stat_tl_from_A, likelihood_score_stats_tl_from_perms, A_perms, t=t, l=l)

def observed_global_subgroup(res: SSResult, A: np.ndarray, subgroup_idx: np.ndarray) -> float:
    return _apply_score(
        res,
        likelihood_score_stat_global_from_A,
        likelihood_score_stats_global_from_perms,
        A,
        subgroup_idx=subgroup_idx,
    )

def perms_global_subgroup(
    res: SSResult,
    A_perms: np.ndarray,
    subgroup_idx: np.ndarray,
) -> np.ndarray:
    return _apply_score(
        res,
        likelihood_score_stat_global_from_A,
        likelihood_score_stats_global_from_perms,
        A_perms,
        subgroup_idx=subgroup_idx,
    )
