"""Shared utilities for the RP experiments."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
Assumption = Literal['static', 'lagged']
Option = Literal['parametric', 'nonparametric']

@dataclass(frozen=True)
class SADDesign:
    T: int
    counts_by_time: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.counts_by_time) != self.T + 1:
            raise ValueError('counts_by_time must have length T+1')
        if any((c < 0 for c in self.counts_by_time)):
            raise ValueError('counts_by_time must be nonnegative')

    @property
    def N(self) -> int:
        return int(np.sum(self.counts_by_time))

    @property
    def pi(self) -> np.ndarray:
        out = np.zeros(self.T + 2, dtype=float)
        out[1:self.T + 2] = np.asarray(self.counts_by_time, dtype=float) / self.N
        return out

    @property
    def pi_leq(self) -> np.ndarray:
        return np.cumsum(self.pi)

    def sample_A(self, rng: np.random.Generator) -> np.ndarray:
        A = np.concatenate([np.full(c, t, dtype=int) for t, c in enumerate(self.counts_by_time, start=1)])
        rng.shuffle(A)
        return A

def shuffle_within(A: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = A.copy()
    out[idx] = rng.permutation(out[idx])
    return out

def _perms_shuffle_subset(
    A_obs: np.ndarray,
    idx: np.ndarray,
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    P = np.empty((n_perms, A_obs.size), dtype=int)
    for r in range(n_perms):
        P[r] = shuffle_within(A_obs, idx, rng)
    return P

def _eligible_tl(
    A_obs: np.ndarray,
    t: int,
    l: int,
    T: int,
    subgroup_idx: Optional[np.ndarray]=None,
) -> np.ndarray:
    if not 1 <= t <= T - l:
        raise ValueError(f'Need 1 <= t <= T-l; got t={t}, l={l}, T={T}')
    idx = np.where((A_obs == t) | (A_obs > t + l))[0]
    return idx if subgroup_idx is None else np.intersect1d(idx, subgroup_idx, assume_unique=False)

def perms_H0(A_obs: np.ndarray, rng: np.random.Generator, n_perms: int) -> np.ndarray:
    P = np.empty((n_perms, A_obs.size), dtype=int)
    for r in range(n_perms):
        P[r] = rng.permutation(A_obs)
    return P

def perms_H_tl(
    A_obs: np.ndarray,
    t: int,
    l: int,
    T: int,
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    return _perms_shuffle_subset(A_obs, _eligible_tl(A_obs, t, l, T), rng, n_perms)

def perms_H_0k(
    A_obs: np.ndarray,
    subgroup_idx: np.ndarray,
    rng: np.random.Generator,
    n_perms: int,
) -> np.ndarray:
    return _perms_shuffle_subset(A_obs, subgroup_idx, rng, n_perms)

# h2 main-DGP constants: sparse positive responders with a small negative majority branch.
# The responder probability is increasing in X_1, so five X_1-quintile
# subgroups have distinct, nonnegative population ATEs from near zero to
# clearly positive.
MAIN_RESPONDER_PROB_SCALE = 0.08
MAIN_POSITIVE_EFFECT = 3.5
MAIN_NEGATIVE_EFFECT = -0.1
MAIN_LAGGED_PROFILE = (1.0, 0.9, 0.81, 0.729, 0.6561)

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def main_responder_probability(X: np.ndarray) -> np.ndarray:
    """Rare-responder probability for the h2 main synthetic DGP."""
    return float(MAIN_RESPONDER_PROB_SCALE) * _sigmoid(np.asarray(X[:, 0], dtype=float))

def generate_covariates(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate covariates for the h2 main synthetic experiments.

    X_1,...,X_4 are iid Uniform[-2,2].  X_5 is a rare positive-responder
    indicator with probability 0.08 sigmoid(X_1).  Units with X_5=1 have a
    large positive CATE, while the majority branch X_5=0 has a small negative
    CATE.  This keeps first moments small but leaves a strong second-moment
    signal for RV/RC.
    """
    X = np.zeros((N, 5), dtype=float)
    X[:, :4] = rng.uniform(-2.0, 2.0, size=(N, 4))
    p = main_responder_probability(X)
    X[:, 4] = rng.binomial(1, p, size=N).astype(float)
    return X

def true_tau_matrix(
    X: np.ndarray,
    T: int,
    assumption: Union[str, int],
    tau_zero: bool,
) -> np.ndarray:
    """True CATE matrix for the h2 main synthetic experiments.

    In the power setting, rare positive responders have effect +3.5 and the
    majority branch has effect -0.10.  The resulting ATE is low because the
    sparse positive branch is partly offset by many small negative effects,
    while E[tau^2] remains large enough for moment-assisted RV/RC.
    """
    if tau_zero:
        return np.zeros((X.shape[0], T), dtype=float)
    base = np.where(np.asarray(X[:, 4], dtype=float) > 0.5, float(MAIN_POSITIVE_EFFECT), float(MAIN_NEGATIVE_EFFECT))
    if assumption == 'static':
        return base[:, None] * np.ones((X.shape[0], T), dtype=float)
    base_profile = np.asarray(MAIN_LAGGED_PROFILE, dtype=float)
    if T <= base_profile.size:
        lag_profile = base_profile[:T].copy()
    else:
        ratio = base_profile[-1] / base_profile[-2] if base_profile.size >= 2 and base_profile[-2] != 0 else 0.72
        extra = base_profile[-1] * (ratio ** np.arange(1, T - base_profile.size + 1, dtype=float))
        lag_profile = np.concatenate([base_profile, extra])
    return base[:, None] * lag_profile[None, :]

def mu0_matrix(X: np.ndarray, T: int) -> np.ndarray:
    return X.sum(axis=1, keepdims=True) * (0.7 + 0.1 * np.arange(1, T + 1))[None, :]

def error_covariance(T: int, assumption: Union[str, int]) -> np.ndarray:
    if assumption == 'static':
        cov = np.zeros((T, T), dtype=float)
        for t in range(T):
            for u in range(T):
                cov[t, u] = 2.0 ** (-abs(t - u))
        return cov
    return np.eye(T)

def generate_errors(
    N: int,
    T: int,
    assumption: Union[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(T), error_covariance(T, assumption), size=N)

def generate_outcomes(
    X: np.ndarray,
    A: np.ndarray,
    T: int,
    assumption: Union[str, int],
    tau_zero: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu0 = mu0_matrix(X, T)
    tau = true_tau_matrix(X, T, assumption, tau_zero)
    eps = generate_errors(X.shape[0], T, assumption, rng)
    Y = mu0 + eps
    for i, a in enumerate(A):
        if a <= T:
            Y[i, a - 1:] += tau[i, :T - a + 1]
    return (Y, mu0, tau)

def make_subgroups_by_quantiles(X: np.ndarray, K: int=4) -> Dict[int, np.ndarray]:
    bins = np.concatenate(([-np.inf], np.quantile(X[:, 0], np.linspace(0, 1, K + 1)[1:-1]), [np.inf]))
    return {k: np.where((X[:, 0] > bins[k - 1]) & (X[:, 0] <= bins[k]))[0] for k in range(1, K + 1)}

# Lighter default GBRTs for the main DGP.
DEFAULT_GBRT_PARAMS: Dict[str, object] = {
    'n_estimators': 400,
    'learning_rate': 0.05,
    'max_depth': 1,
    'subsample': 0.8,
    'min_samples_leaf': 10,
}

def default_mu_gbrt_params(assumption: Optional[Union[str, int]]=None) -> Dict[str, object]:
    _ = assumption
    return dict(DEFAULT_GBRT_PARAMS)

def make_regressor(
    *,
    degree: int=1,
    ridge_alpha: float=0.001,
    option: str='parametric',
    nonparametric_kind: str='gbrt',
    random_state: Optional[int]=12345,
    gbrt_params: Optional[Dict[str, object]]=None,
):
    if option == 'parametric':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        return Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                         ('scaler', StandardScaler()),
                         ('ridge', Ridge(alpha=ridge_alpha, fit_intercept=True))])
    kind = str(nonparametric_kind).strip().lower()
    if kind in {'gbrt', 'boosting', 'histgb', 'histgradientboosting'}:
        from sklearn.ensemble import GradientBoostingRegressor
        params = {**DEFAULT_GBRT_PARAMS, **(gbrt_params or {})}
        params.setdefault('random_state', random_state)
        return GradientBoostingRegressor(**params)
    if kind in {'decision_tree', 'decisiontree', 'tree', 'cart'}:
        from sklearn.tree import DecisionTreeRegressor
        params = dict(gbrt_params or {})
        params.setdefault('random_state', random_state)
        return DecisionTreeRegressor(**params)
    raise ValueError(f'Unknown nonparametric_kind={nonparametric_kind!r}. Supported kinds are gradient boosting and decision tree.')

def fit_mu_models(
    X: np.ndarray,
    Y: np.ndarray,
    degree: int=1,
    ridge_alpha: float=0.001,
    *,
    option: str='parametric',
    nonparametric_kind: str='gbrt',
    random_state: Optional[int]=12345,
    gbrt_params: Optional[Dict[str, object]]=None,
    assumption: Optional[Union[str, int]]=None,
) -> List:
    mu_gbrt_params = gbrt_params
    if option == 'nonparametric' and mu_gbrt_params is None:
        mu_gbrt_params = default_mu_gbrt_params(assumption)
    models = []
    for t in range(Y.shape[1]):
        rs = None if random_state is None else int(random_state) + t
        m = make_regressor(
            degree=degree,
            ridge_alpha=ridge_alpha,
            option=option,
            nonparametric_kind=nonparametric_kind,
            random_state=rs,
            gbrt_params=mu_gbrt_params,
        )
        m.fit(X, Y[:, t])
        models.append(m)
    return models

def predict_mu(models: List, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models]) if models else np.empty((X.shape[0], 0), dtype=float)

class FixedPredictionRegressor:
    """Regressor-like wrapper that returns a pre-tabulated prediction vector.

    This is used for oracle nuisance experiments in the simulator. It supports
    prediction on the original X or on exact row subsets of that X.
    """

    def __init__(self, X_ref: np.ndarray, y_ref: np.ndarray):
        self.X_ref = np.asarray(X_ref, dtype=float)
        self.y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
        if self.X_ref.ndim != 2 or self.y_ref.shape[0] != self.X_ref.shape[0]:
            raise ValueError('X_ref and y_ref must have compatible shapes')
        self._lookup = {np.ascontiguousarray(self.X_ref[i]).tobytes(): i for i in range(self.X_ref.shape[0])}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_ref.shape[1]:
            raise ValueError('X must have the same number of columns as X_ref')
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            key = np.ascontiguousarray(X[i]).tobytes()
            if key not in self._lookup:
                raise ValueError('FixedPredictionRegressor only supports exact rows from the original X')
            out[i] = self.y_ref[self._lookup[key]]
        return out

def wrap_fixed_mu_models(X: np.ndarray, mu_hat: np.ndarray) -> List[FixedPredictionRegressor]:
    mu_hat = np.asarray(mu_hat, dtype=float)
    if mu_hat.ndim != 2 or mu_hat.shape[0] != X.shape[0]:
        raise ValueError('mu_hat must have shape (N, T)')
    return [FixedPredictionRegressor(X, mu_hat[:, t]) for t in range(mu_hat.shape[1])]

def true_observed_mean_matrix(
    mu0_true: np.ndarray,
    tau_true: np.ndarray,
    design: SADDesign,
) -> np.ndarray:
    """Return E[Y | X] under the known randomized adoption design."""
    mu0_true = np.asarray(mu0_true, dtype=float)
    tau_true = np.asarray(tau_true, dtype=float)
    if mu0_true.shape != tau_true.shape:
        raise ValueError('mu0_true and tau_true must have the same shape')
    T = design.T
    mu = mu0_true.copy()
    for u in range(1, T + 1):
        for a in range(1, u + 1):
            mu[:, u - 1] += float(design.pi[a]) * tau_true[:, u - a]
    return mu

def true_ss_mhat_dict(tau_true: np.ndarray, design: SADDesign) -> Dict[Tuple[int, int], np.ndarray]:
    """Oracle conditional means for the sample-splitting second-stage regressions."""
    tau_true = np.asarray(tau_true, dtype=float)
    T = design.T
    out: Dict[Tuple[int, int], np.ndarray] = {}
    delta = np.zeros_like(tau_true, dtype=float)
    for t in range(1, T + 1):
        for a in range(1, t + 1):
            delta[:, t - 1] += float(design.pi[a]) * tau_true[:, t - a]
    for t in range(1, T + 1):
        for k in range(t):
            a = t - k
            out[t, k] = float(design.pi[a]) * (tau_true[:, k] - delta[:, t - 1])
    return out
true_me_mhat_dict = true_ss_mhat_dict

def true_rv_second_moments(
    tau_true: np.ndarray,
    design: SADDesign,
    assumption: Union[str, int],
) -> np.ndarray:
    """Oracle b_t(x) = E[R_t^2 | X=x] for RV after oracle mean residualization."""
    tau_true = np.asarray(tau_true, dtype=float)
    T = design.T
    diag_var = np.diag(error_covariance(T, assumption)).astype(float)
    delta = np.zeros_like(tau_true, dtype=float)
    second = np.zeros_like(tau_true, dtype=float)
    for t in range(1, T + 1):
        for a in range(1, t + 1):
            w = float(design.pi[a])
            lag = t - a
            delta[:, t - 1] += w * tau_true[:, lag]
            second[:, t - 1] += w * tau_true[:, lag] ** 2
        second[:, t - 1] = diag_var[t - 1] + second[:, t - 1] - delta[:, t - 1] ** 2
    return second

def true_rc_offdiag_moments(
    tau_true: np.ndarray,
    design: SADDesign,
    assumption: Union[str, int],
) -> np.ndarray:
    """Oracle m_{t,s}(x) = E[R_t R_s | X=x] for the off-diagonal RC moments."""
    tau_true = np.asarray(tau_true, dtype=float)
    T = design.T
    N = tau_true.shape[0]
    err_cov = error_covariance(T, assumption)
    delta = np.zeros_like(tau_true, dtype=float)
    for t in range(1, T + 1):
        for a in range(1, t + 1):
            delta[:, t - 1] += float(design.pi[a]) * tau_true[:, t - a]
    pairs = [(t, s) for t in range(2, T + 1) for s in range(1, t)]
    out = np.zeros((N, len(pairs)), dtype=float)
    for j, (t, s) in enumerate(pairs):
        cross = np.zeros(N, dtype=float)
        for a in range(1, min(t, s) + 1):
            cross += float(design.pi[a]) * tau_true[:, t - a] * tau_true[:, s - a]
        out[:, j] = float(err_cov[t - 1, s - 1]) + cross - delta[:, t - 1] * delta[:, s - 1]
    return out

def poly_basis_transformer(degree: int=1):
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(degree=degree, include_bias=True)



def resolve_unit_signs_from_residual_trajectory(
    design: SADDesign,
    residuals: np.ndarray,
    tau_pm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the +/- orientation of an up-to-sign CATE estimate unit by unit.

    For each unit i and each candidate latent start time a in {1, ..., T+1},
    this assignment-free stage builds

        q_i(a)_t = sum_{l=0}^{t-1} (1{a=t-l} - pi_{t-l}) tau_pm_{i,l},

    and chooses the sign s in {-1,+1} and candidate a that minimize
    sum_t (R_{i,t} - s q_i(a)_t)^2.  The observed realized assignment A_i is
    not used anywhere in this computation.
    """
    R = np.asarray(residuals, dtype=float)
    tau = np.asarray(tau_pm, dtype=float)
    if R.ndim != 2:
        raise ValueError('residuals must have shape (N, T)')
    if tau.ndim != 2:
        raise ValueError('tau_pm must have shape (N, T)')
    if R.shape != tau.shape:
        raise ValueError(f'residuals and tau_pm must have the same shape; got {R.shape} and {tau.shape}')
    N, T = R.shape
    if int(design.T) != T:
        raise ValueError(f'design.T={design.T} is incompatible with arrays with T={T}')
    pi = np.asarray(design.pi, dtype=float)
    G = np.zeros((T + 1, T, T), dtype=float)
    for a_idx, a in enumerate(range(1, T + 2)):
        for t in range(1, T + 1):
            for l in range(0, t):
                cohort = t - l
                G[a_idx, t - 1, l] = (1.0 if a == cohort else 0.0) - float(pi[cohort])
    q = np.einsum('atl,nl->nat', G, tau)
    score = np.einsum('nt,nat->na', R, q)
    norm_R = np.sum(R ** 2, axis=1, keepdims=True)
    norm_q = np.sum(q ** 2, axis=2)
    losses = norm_R + norm_q - 2.0 * np.abs(score)
    safe_losses = np.nan_to_num(losses, nan=np.inf, posinf=np.inf, neginf=np.inf)
    best_idx = np.argmin(safe_losses, axis=1)
    rows = np.arange(N)
    best_score = score[rows, best_idx]
    sign_hat = np.where(best_score >= 0.0, 1.0, -1.0).astype(float)
    finite_best = np.isfinite(safe_losses[rows, best_idx])
    sign_hat[~finite_best] = 1.0
    best_assignment = best_idx.astype(int) + 1
    best_loss = safe_losses[rows, best_idx]
    return (sign_hat[:, None] * tau, sign_hat, best_assignment, best_loss)

@dataclass
class LikelihoodScoreComponents:
    T: int
    Y: np.ndarray
    tau_hat: np.ndarray
    mu0_hat: np.ndarray

def build_likelihood_score_components(
    design: SADDesign,
    X: np.ndarray,
    Y: np.ndarray,
    mu_models: List,
    tau_hat: np.ndarray,
) -> LikelihoodScoreComponents:
    mu_hat = predict_mu(mu_models, X)
    mu0_hat = np.zeros_like(mu_hat)
    for u in range(1, design.T + 1):
        mu0_hat[:, u - 1] = mu_hat[:, u - 1] - np.sum(design.pi[1:u + 1][::-1] * tau_hat[:, :u], axis=1)
    return LikelihoodScoreComponents(T=design.T, Y=Y, tau_hat=tau_hat, mu0_hat=mu0_hat)

@dataclass
class AIPWScoreComponents:
    T: int
    Y: np.ndarray
    tau_hat: np.ndarray
    mu0_hat: np.ndarray
    pi: np.ndarray
    pi_leq: np.ndarray

def build_aipw_score_components(
    design: SADDesign,
    X: np.ndarray,
    Y: np.ndarray,
    mu_models: List,
    tau_hat: np.ndarray,
) -> AIPWScoreComponents:
    comps = build_likelihood_score_components(design, X, Y, mu_models, tau_hat)
    return AIPWScoreComponents(
        T=design.T,
        Y=np.asarray(Y, dtype=float),
        tau_hat=np.asarray(tau_hat, dtype=float),
        mu0_hat=np.asarray(comps.mu0_hat, dtype=float),
        pi=np.asarray(design.pi, dtype=float),
        pi_leq=np.asarray(design.pi_leq, dtype=float),
    )

def _subset(arr: np.ndarray, subset: Optional[np.ndarray]) -> np.ndarray:
    return arr if subset is None else arr[subset]

def _score_terms(
    comps: LikelihoodScoreComponents,
    u: int,
    l: int,
    subset: Optional[np.ndarray],
) -> np.ndarray:
    idx = slice(None) if subset is None else subset
    return comps.tau_hat[idx, l] * (comps.Y[idx, u - 1] - comps.mu0_hat[idx, u - 1]) - 0.5 * comps.tau_hat[idx, l] ** 2

def likelihood_score_stat_tl_from_A(
    A: np.ndarray,
    comps: LikelihoodScoreComponents,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    if not 1 <= t <= comps.T - l:
        raise ValueError('t must satisfy 1 <= t <= T-l')
    A_sub = _subset(A, subset)
    return float(np.sum((A_sub == t) * _score_terms(comps, t + l, l, subset)))

def likelihood_score_stats_tl_from_perms(
    A_perms: np.ndarray,
    comps: LikelihoodScoreComponents,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    if not 1 <= t <= comps.T - l:
        raise ValueError('t must satisfy 1 <= t <= T-l')
    A_sub = A_perms if subset is None else A_perms[:, subset]
    return ((A_sub == t).astype(float) @ _score_terms(comps, t + l, l, subset)).astype(float)

def likelihood_score_stat_lag_from_A(
    A: np.ndarray,
    comps: LikelihoodScoreComponents,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    if not 0 <= l <= comps.T - 1:
        raise ValueError('l must satisfy 0 <= l <= T-1')
    return float(np.mean([likelihood_score_stat_tl_from_A(A, comps, t=t, l=l, subset=subset) for t in range(1, comps.T - l + 1)]))

def likelihood_score_stats_lag_from_perms(
    A_perms: np.ndarray,
    comps: LikelihoodScoreComponents,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    if not 0 <= l <= comps.T - 1:
        raise ValueError('l must satisfy 0 <= l <= T-1')
    stats = sum((likelihood_score_stats_tl_from_perms(A_perms, comps, t=t, l=l, subset=subset) for t in range(1, comps.T - l + 1)))
    return (stats / (comps.T - l)).astype(float)

def likelihood_score_stat_global_from_A(
    A: np.ndarray,
    comps: LikelihoodScoreComponents,
    subset: Optional[np.ndarray]=None,
) -> float:
    return float(np.mean([likelihood_score_stat_lag_from_A(A, comps, l=l, subset=subset) for l in range(comps.T)]))

def likelihood_score_stats_global_from_perms(
    A_perms: np.ndarray,
    comps: LikelihoodScoreComponents,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    return (sum((likelihood_score_stats_lag_from_perms(A_perms, comps, l=l, subset=subset) for l in range(comps.T))) / comps.T).astype(float)

def aipw_score_stat_tl_from_A(
    A: np.ndarray,
    comps: AIPWScoreComponents,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    if not 1 <= t <= comps.T - l:
        raise ValueError('t must satisfy 1 <= t <= T-l')
    A_sub = _subset(np.asarray(A, dtype=int), subset)
    Y_sub = _subset(np.asarray(comps.Y, dtype=float), subset)
    tau = _subset(np.asarray(comps.tau_hat, dtype=float), subset)[:, l]
    mu0 = _subset(np.asarray(comps.mu0_hat, dtype=float), subset)[:, t + l - 1]
    if tau.size == 0:
        return 0.0
    out = tau.copy()
    if comps.pi[t] > 0.0:
        out += (A_sub == t).astype(float) * ((Y_sub[:, t + l - 1] - (mu0 + tau)) / comps.pi[t])
    denom = 1.0 - comps.pi_leq[t + l]
    if denom > 0.0:
        out -= (A_sub > t + l).astype(float) * ((Y_sub[:, t + l - 1] - mu0) / denom)
    return float(np.mean(out))

def aipw_score_stats_tl_from_perms(
    A_perms: np.ndarray,
    comps: AIPWScoreComponents,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    if not 1 <= t <= comps.T - l:
        raise ValueError('t must satisfy 1 <= t <= T-l')
    A_sub = np.asarray(A_perms, dtype=int) if subset is None else np.asarray(A_perms, dtype=int)[:, subset]
    Y_sub = _subset(np.asarray(comps.Y, dtype=float), subset)
    tau = _subset(np.asarray(comps.tau_hat, dtype=float), subset)[:, l]
    mu0 = _subset(np.asarray(comps.mu0_hat, dtype=float), subset)[:, t + l - 1]
    if tau.size == 0:
        return np.zeros(A_sub.shape[0], dtype=float)
    out = np.full(A_sub.shape[0], float(np.mean(tau)), dtype=float)
    if comps.pi[t] > 0.0:
        out += (A_sub == t).astype(float) @ (Y_sub[:, t + l - 1] - (mu0 + tau)) / (comps.pi[t] * tau.size)
    denom = 1.0 - comps.pi_leq[t + l]
    if denom > 0.0:
        out -= (A_sub > t + l).astype(float) @ (Y_sub[:, t + l - 1] - mu0) / (denom * tau.size)
    return out

def aipw_score_stat_lag_from_A(
    A: np.ndarray,
    comps: AIPWScoreComponents,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    if not 0 <= l <= comps.T - 1:
        raise ValueError('l must satisfy 0 <= l <= T-1')
    return float(np.mean([aipw_score_stat_tl_from_A(A, comps, t=t, l=l, subset=subset) for t in range(1, comps.T - l + 1)]))

def aipw_score_stats_lag_from_perms(
    A_perms: np.ndarray,
    comps: AIPWScoreComponents,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    if not 0 <= l <= comps.T - 1:
        raise ValueError('l must satisfy 0 <= l <= T-1')
    stats = sum((aipw_score_stats_tl_from_perms(A_perms, comps, t=t, l=l, subset=subset) for t in range(1, comps.T - l + 1)))
    return (stats / (comps.T - l)).astype(float)

def aipw_score_stat_global_from_A(
    A: np.ndarray,
    comps: AIPWScoreComponents,
    subset: Optional[np.ndarray]=None,
) -> float:
    return float(np.mean([aipw_score_stat_lag_from_A(A, comps, l=l, subset=subset) for l in range(comps.T)]))

def aipw_score_stats_global_from_perms(
    A_perms: np.ndarray,
    comps: AIPWScoreComponents,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    return (sum((aipw_score_stats_lag_from_perms(A_perms, comps, l=l, subset=subset) for l in range(comps.T))) / comps.T).astype(float)

def dm_stat_tl_from_A(
    A: np.ndarray,
    Y: np.ndarray,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    u = t + l
    A_sub, Y_sub = (_subset(A, subset), _subset(Y, subset))
    treated, ctrl = (A_sub == t, A_sub > u)
    if treated.sum() == 0 or ctrl.sum() == 0:
        return 0.0
    return float(Y_sub[treated, u - 1].mean() - Y_sub[ctrl, u - 1].mean())

def dm_stats_tl_from_perms(
    A_perms: np.ndarray,
    Y: np.ndarray,
    t: int,
    l: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    u = t + l
    A_sub = A_perms if subset is None else A_perms[:, subset]
    Y_u = Y[:, u - 1] if subset is None else Y[subset, u - 1]
    treated = (A_sub == t).astype(float)
    ctrl = (A_sub > u).astype(float)
    sum_t, sum_c = (treated @ Y_u, ctrl @ Y_u)
    n_t, n_c = (treated.sum(axis=1), ctrl.sum(axis=1))
    mean_t = np.divide(sum_t, n_t, out=np.zeros_like(sum_t), where=n_t > 0)
    mean_c = np.divide(sum_c, n_c, out=np.zeros_like(sum_c), where=n_c > 0)
    return (mean_t - mean_c).astype(float)

def dm_stat_global_from_A(
    A: np.ndarray,
    Y: np.ndarray,
    T: int,
    subset: Optional[np.ndarray]=None,
) -> float:
    return float(np.mean([dm_stat_tl_from_A(A, Y, t=t, l=l, subset=subset) for l in range(T) for t in range(1, T - l + 1)]))

def dm_stats_global_from_perms(
    A_perms: np.ndarray,
    Y: np.ndarray,
    T: int,
    subset: Optional[np.ndarray]=None,
) -> np.ndarray:
    n = sum((T - l for l in range(T)))
    return (sum((dm_stats_tl_from_perms(A_perms, Y, t=t, l=l, subset=subset) for l in range(T) for t in range(1, T - l + 1))) / n).astype(float)

def build_true_oracle_score_components(
    design: SADDesign,
    Y: np.ndarray,
    mu0_true: np.ndarray,
    tau_true: np.ndarray,
) -> LikelihoodScoreComponents:
    return LikelihoodScoreComponents(T=design.T, Y=Y, tau_hat=tau_true, mu0_hat=mu0_true)

def right_tailed_pvalue(stat_obs: float, stats_perm: np.ndarray) -> float:
    return float((1.0 + np.sum(stats_perm >= stat_obs)) / (len(stats_perm) + 1.0))
true_rcd_second_moments = true_rv_second_moments
true_rco_offdiag_moments = true_rc_offdiag_moments
