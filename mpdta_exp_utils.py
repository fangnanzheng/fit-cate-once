from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.special import stdtr
import helpers
import rc
Assumption = Literal['static', 'lagged']
Statistic = Literal['aipw', 'likelihood']
NonparametricKind = Literal['gbrt', 'cart']
REQUIRED_COLUMNS = ['year', 'countyreal', 'lpop', 'lemp', 'first.treat', 'treat']
REPORTED_LAGS = (0, 1, 2, 3, 4)
TWFE_EVENT_LAGS = (0, 1, 2, 3)
VALID_ASSUMPTIONS = ('static', 'lagged')
VALID_STATISTICS = ('aipw', 'likelihood')
VALID_NONPARAMETRIC_KINDS = ('gbrt', 'cart')
TREATED_FRAC_SUBGROUP = 0.65
# h5 MPDTA validity/power DGP calibration.  Treatment timing remains complete
# randomization with fixed cohort counts, and the h3/h4 untreated trend is
# retained.  The power treatment effects are now smooth negative CATEs in the
# empirical rank of standardized lpop, avoiding the h4 top-tail spike.
MPDTA_TREND_SCALE = 2.0
SHARED_MU_LINEAR = MPDTA_TREND_SCALE * np.array([-0.9, -0.3, 0.15, 0.75, 1.35], dtype=float)
SHARED_MU_QUADRATIC = MPDTA_TREND_SCALE * np.array([-0.25, -0.08, 0.08, 0.30, 0.55], dtype=float)
SHARED_MU_CUBIC = MPDTA_TREND_SCALE * np.array([-0.12, -0.03, 0.03, 0.12, 0.28], dtype=float)
STATIC_POWER_BASE_MAGNITUDE = 0.40
STATIC_POWER_RANK_SLOPE = 0.04
LAGGED_POWER_BASE_PROFILE = (0.6,1.0,1.4,1.8,2.2)
LAGGED_POWER_RANK_SLOPE_PROFILE = (0.4,0.9,1.4,1.9,2.4)
RIDGE_ALPHA = 0.001
SUBGROUP_GBRT_DEFAULTS = {
    'mu': {'max_depth': 1, 'min_samples_leaf': 20},
    'b': {'max_depth': 1, 'min_samples_leaf': 1},
}
# MPDTA-specific GBRT settings.  The untreated mean contains a nontrivial
# cubic term, so the outcome regressions need more flexibility than the global
# synthetic-experiment defaults.  The h5 power CATEs are smooth in one
# covariate, so the residual-moment regressions are deliberately shallow and
# conservative rather than tuned to chase noisy residual products.
MPDTA_MU_GBRT_PARAMS = {
    "n_estimators": 2500,
    "learning_rate": 0.01,
    "max_depth": 2,
    "subsample": 1.0,
    "min_samples_leaf": 6,
}
MPDTA_RV_MOMENT_GBRT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.01,
    "max_depth": 1,
    "subsample": 1.0,
    "min_samples_leaf": 60,
    "loss": "huber",
    "alpha": 0.90,
}
MPDTA_RC_MOMENT_GBRT_PARAMS = {
    "n_estimators": 450,
    "learning_rate": 0.008,
    "max_depth": 1,
    "subsample": 1.0,
    "min_samples_leaf": 60,
    "loss": "huber",
    "alpha": 0.90,
}

@dataclass(frozen=True)
class MPDTAData:
    unit_ids: List[int]
    years: List[int]
    X: np.ndarray
    X_raw: pd.DataFrame
    Y_obs: np.ndarray
    gating_variable: str = 'lpop'

@dataclass(frozen=True)
class TestResult:
    method: str
    pvalue: float
    stat: float

@dataclass(frozen=True)
class NuisanceFit:
    tau_hat: np.ndarray
    mu_models: List
    comps: object
    fit_result: Optional[object] = None

@dataclass(frozen=True)
class SubgroupFocalModel:
    mu_models: List
    b_models: List

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - float(np.mean(x))) / max(float(np.std(x)), 1e-12)

STATISTIC_LABEL = 'likelihood'

def score_label() -> str:
    return STATISTIC_LABEL

def normalize_statistic(statistic: str) -> Statistic:
    value = str(statistic).strip().lower()
    if value not in VALID_STATISTICS:
        raise ValueError(f'statistic must be one of {VALID_STATISTICS}; got {statistic!r}')
    return value

def normalize_nonparametric_kind(kind: str) -> NonparametricKind:
    value = str(kind).strip().lower()
    if value == 'decision_tree':
        value = 'cart'
    if value not in VALID_NONPARAMETRIC_KINDS:
        raise ValueError(f'nonparametric_kind must be one of {VALID_NONPARAMETRIC_KINDS}; got {kind!r}')
    return value

def build_score_components(
    design: helpers.SADDesign,
    X: np.ndarray,
    Y: np.ndarray,
    mu_models: List,
    tau_hat: np.ndarray,
    *,
    statistic: Statistic='likelihood',
):
    statistic = normalize_statistic(statistic)
    if statistic == 'aipw':
        return helpers.build_aipw_score_components(design, X, Y, mu_models, tau_hat)
    return helpers.build_likelihood_score_components(design, X, Y, mu_models, tau_hat)

def load_mpdta_data(data_path: str | Path) -> MPDTAData:
    df = pd.read_csv(Path(data_path), usecols=REQUIRED_COLUMNS).copy()
    df[['countyreal', 'year']] = df[['countyreal', 'year']].astype(int)
    df = df.sort_values(['countyreal', 'year']).reset_index(drop=True)
    years = sorted(df['year'].unique().tolist())
    unit_ids = sorted(df['countyreal'].unique().tolist())
    if not bool((df.groupby('countyreal')['year'].nunique() == len(years)).all()):
        raise ValueError('mpdta must be a balanced panel.')
    unit_df = df.drop_duplicates('countyreal').set_index('countyreal').loc[unit_ids]
    y = df.pivot(index='countyreal', columns='year', values='lemp').loc[unit_ids, years].to_numpy(dtype=float)
    lpop = unit_df['lpop'].to_numpy(dtype=float)
    x_raw = pd.DataFrame({'lpop': lpop}, index=pd.Index(unit_ids, name='countyreal'))
    return MPDTAData(unit_ids, years, _zscore(lpop)[:, None], x_raw, y)

def _treated_counts(n_units: int, n_periods: int, treated_frac: float) -> List[int]:
    treat_total = min(max(int(round(treated_frac * n_units)), n_periods), n_units - 1)
    raw = np.full(n_periods, treat_total / max(n_periods, 1), dtype=float)
    counts = np.floor(raw).astype(int)
    counts[:int(treat_total - counts.sum())] += 1
    return counts.tolist()

def sample_design_A(
    data: MPDTAData,
    rng: np.random.Generator,
    treated_frac: float,
) -> Tuple[helpers.SADDesign, np.ndarray]:
    counts = _treated_counts(len(data.unit_ids), len(data.years), treated_frac)
    never = len(data.unit_ids) - sum(counts)
    if never <= 0:
        counts[-1] -= 1
        never = 1
    design = helpers.SADDesign(T=len(data.years), counts_by_time=counts + [never])
    return (design, design.sample_A(rng))

def _counts_from_shares(n_units: int, shares: Sequence[float]) -> List[int]:
    raw = np.asarray(shares, dtype=float)
    if raw.ndim != 1 or np.any(raw <= 0.0):
        raise ValueError('shares must be a positive 1D sequence')
    raw = raw / float(np.sum(raw))
    counts = np.floor(n_units * raw).astype(int)
    counts[np.argmax(raw)] += int(n_units - counts.sum())
    return counts.astype(int).tolist()

def sparse_lag_counts(n_units: int, n_periods: int, *, assumption: str='static') -> List[int]:
    """Cohort counts for MPDTA lag experiments.

    The semi-synthetic lag experiments use the same fixed cohort proportions
    as the main and consistency experiments: treated-start cohorts have
    proportions (0.1, 0.2, 0.2, 0.2, 0.2) and the never-treated cohort has
    proportion 0.1.
    """
    if n_periods != 5:
        raise ValueError(f'MPDTA lag DGP is implemented for T=5; got T={n_periods}')
    return _counts_from_shares(n_units, [0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

def sample_sparse_lag_design_A(
    data: MPDTAData,
    rng: np.random.Generator,
    *,
    assumption: str='static',
) -> Tuple[helpers.SADDesign, np.ndarray]:
    design = helpers.SADDesign(
        T=len(data.years),
        counts_by_time=sparse_lag_counts(len(data.unit_ids), len(data.years), assumption=assumption),
    )
    return (design, design.sample_A(rng))

def default_noise_scale(experiment: str, *, assumption: Optional[str]=None) -> float:
    if experiment == 'subgroup':
        return 0.04
    if experiment not in {'validity', 'power'}:
        raise ValueError("experiment must be 'validity', 'power', or 'subgroup'")
    if assumption == 'lagged':
        return 0.3
    return 0.2

def _noise(
    shape: Tuple[int, int],
    scale: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    eps = rng.normal(size=shape)
    eps -= eps.mean(axis=0, keepdims=True)
    eps /= max(float(np.std(eps)), 1e-08)
    return noise_scale * scale * eps

def _trend_features(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    return (z, z ** 2 - 1.0, z ** 3 - 3.0 * z)

def _structured_errors(
    n_units: int,
    n_periods: int,
    *,
    assumption: str,
    scale: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cov = (float(noise_scale) * float(scale)) ** 2 * helpers.error_covariance(n_periods, assumption)
    return rng.multivariate_normal(np.zeros(n_periods, dtype=float), cov, size=n_units)

def build_lag_mu0_mean_matrix(data: MPDTAData, *, assumption: str='static') -> np.ndarray:
    """Shared nonlinear untreated mean for MPDTA lag experiments.

    The same nonlinear baseline is used for static and lagged experiments so
    that TWFE misspecification comes from the untreated trend rather than from
    a treatment-effect construction tailored separately to each assumption.
    """
    z = np.asarray(data.X[:, 0], dtype=float)
    h1, h2, h3 = _trend_features(z)
    mu0 = np.repeat(data.Y_obs.mean(axis=0, keepdims=True), len(data.unit_ids), axis=0)
    y_scale = float(np.std(data.Y_obs))
    return mu0 + y_scale * (
        np.outer(h1, SHARED_MU_LINEAR)
        + np.outer(h2, SHARED_MU_QUADRATIC)
        + np.outer(h3, SHARED_MU_CUBIC)
    )

def build_mu_matrix(
    data: MPDTAData,
    *,
    noise_scale: float,
    rng: np.random.Generator,
    assumption: str='static',
) -> np.ndarray:
    y_scale = float(np.std(data.Y_obs))
    mu0 = build_lag_mu0_mean_matrix(data, assumption=assumption)
    return mu0 + _structured_errors(len(data.unit_ids), len(data.years), assumption=assumption, scale=y_scale, noise_scale=noise_scale, rng=rng)


def mpdta_error_covariance(data: MPDTAData, *, assumption: str, noise_scale: float) -> np.ndarray:
    y_scale = float(np.std(data.Y_obs))
    return (float(noise_scale) * y_scale) ** 2 * helpers.error_covariance(len(data.years), assumption)

def true_lag_observed_mean_matrix(
    data: MPDTAData,
    design: helpers.SADDesign,
    tau_true: np.ndarray,
    *,
    assumption: str,
) -> np.ndarray:
    mu0_mean = build_lag_mu0_mean_matrix(data, assumption=assumption)
    return helpers.true_observed_mean_matrix(mu0_mean, tau_true, design)

def true_rv_second_moments_mpdta(
    data: MPDTAData,
    tau_true: np.ndarray,
    design: helpers.SADDesign,
    *,
    assumption: str,
    noise_scale: float,
) -> np.ndarray:
    tau_true = np.asarray(tau_true, dtype=float)
    T = design.T
    err_diag = np.diag(mpdta_error_covariance(data, assumption=assumption, noise_scale=noise_scale)).astype(float)
    delta = np.zeros_like(tau_true, dtype=float)
    second = np.zeros_like(tau_true, dtype=float)
    for t in range(1, T + 1):
        for a in range(1, t + 1):
            w = float(design.pi[a])
            lag = t - a
            delta[:, t - 1] += w * tau_true[:, lag]
            second[:, t - 1] += w * tau_true[:, lag] ** 2
        second[:, t - 1] = err_diag[t - 1] + second[:, t - 1] - delta[:, t - 1] ** 2
    return second

def true_rc_offdiag_moments_mpdta(
    data: MPDTAData,
    tau_true: np.ndarray,
    design: helpers.SADDesign,
    *,
    assumption: str,
    noise_scale: float,
) -> np.ndarray:
    tau_true = np.asarray(tau_true, dtype=float)
    T = design.T
    N = tau_true.shape[0]
    err_cov = mpdta_error_covariance(data, assumption=assumption, noise_scale=noise_scale)
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

def build_subgroup_mu_matrix(
    data: MPDTAData,
    *,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.repeat(data.Y_obs.mean(axis=0, keepdims=True), len(data.unit_ids), axis=0)
    return mu + _noise((len(data.unit_ids), len(data.years)), float(np.std(data.Y_obs)), noise_scale, rng)

def _centered_rank_power_score(data: MPDTAData) -> np.ndarray:
    """Smooth rank score W_i in [-1, 1] based on standardized lpop.

    W_i = 2 * ((rank_i - 0.5) / N) - 1, using average ranks for ties.
    The h5 power DGP uses W_i to create a normal, all-negative CATE rather
    than a sparse top-tail jump.
    """
    z = np.asarray(data.X[:, 0], dtype=float)
    n = max(int(z.size), 1)
    ranks = pd.Series(z).rank(method='average').to_numpy(dtype=float)
    return 2.0 * ((ranks - 0.5) / float(n)) - 1.0

def _extend_profile(values: Sequence[float], n_periods: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if n_periods <= values.size:
        return values[:n_periods].copy()
    if values.size < 2 or abs(float(values[-2])) <= 1e-12:
        ratio = 1.0
    else:
        ratio = float(values[-1] / values[-2])
    extra = values[-1] * (ratio ** np.arange(1, n_periods - values.size + 1, dtype=float))
    return np.concatenate([values, extra])

def build_lag_tau_matrix(
    data: MPDTAData,
    *,
    assumption: str,
    setting: str,
    validity_null_lag: Optional[int]=None,
    effect_size: Optional[float]=None,
) -> np.ndarray:
    n_units = len(data.unit_ids)
    n_periods = len(data.years)
    setting = str(setting).strip().lower()
    if setting == 'validity':
        # MPDTA validity experiments use the sharp all-zero null. This keeps the
        # validity exercise a size check; TWFE over-rejection must come from
        # finite-sample behavior under strong untreated trends, not hidden
        # nonzero effects.
        return np.zeros((n_units, n_periods), dtype=float)

    rank_score = _centered_rank_power_score(data)
    if assumption == 'static':
        base = float(STATIC_POWER_BASE_MAGNITUDE)
        slope = float(STATIC_POWER_RANK_SLOPE if effect_size is None else effect_size)
        # All effects are negative; higher-lpop ranks have slightly larger magnitude.
        tau_static = -(base + slope * rank_score)
        return np.repeat(tau_static[:, None], n_periods, axis=1)

    base_profile = _extend_profile(LAGGED_POWER_BASE_PROFILE, n_periods)
    slope_profile = _extend_profile(LAGGED_POWER_RANK_SLOPE_PROFILE, n_periods)
    if effect_size is not None:
        # Optional scale for quick calibration experiments.  The h5 default uses
        # the smooth-rank profile constants above.
        slope_profile = float(effect_size) * slope_profile
    return -(base_profile[None, :] + rank_score[:, None] * slope_profile[None, :])

def apply_tau(Y0: np.ndarray, A: np.ndarray, tau: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y0, dtype=float).copy()
    T = Y.shape[1]
    for i, a in enumerate(np.asarray(A, dtype=int)):
        if 1 <= a <= T:
            Y[i, a - 1:] += tau[i, :T - a + 1]
    return Y

def estimate_lag_nuisance(
    design: helpers.SADDesign,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    assumption: str,
    random_state: int,
    rv_mu_degree: int=2,
    rv_b_degree: int=2,
    rc_mu_degree: int=5,
    rc_basis_degree: int=2,
    ridge_alpha: float=RIDGE_ALPHA,
    rc_lambda_trace: float=0.01,
    rc_nls_ridge: float=0.01,
    true_mu: Optional[np.ndarray]=None,
    true_moment: Optional[np.ndarray]=None,
) -> NuisanceFit:
    if assumption == 'static':
        fit = rc.fit_rv(
            X,
            Y,
            design,
            mu_degree=int(rv_mu_degree),
            b_degree=int(rv_b_degree),
            ridge_alpha=float(ridge_alpha),
            option='nonparametric',
            assumption='static',
            mu_gbrt_params=dict(MPDTA_MU_GBRT_PARAMS),
            b_gbrt_params=dict(MPDTA_RV_MOMENT_GBRT_PARAMS),
            random_state=random_state,
            true_mu=true_mu,
            true_b=true_moment,
        )
    else:
        fit = rc.fit_rc(
            X,
            Y,
            design,
            mu_degree=int(rc_mu_degree),
            basis_degree=int(rc_basis_degree),
            ridge_alpha=float(ridge_alpha),
            option='nonparametric',
            assumption='lagged',
            mu_gbrt_params=dict(MPDTA_MU_GBRT_PARAMS),
            moment_gbrt_params=dict(MPDTA_RC_MOMENT_GBRT_PARAMS),
            random_state=random_state,
            lambda_trace=float(rc_lambda_trace),
            use_convex_init=True,
            refine_nls=True,
            n_starts=1,
            max_nfev=200,
            nls_ridge=float(rc_nls_ridge),
            true_mu=true_mu,
            true_moment_target=true_moment,
        )
    tau_hat = np.asarray(fit.tau_hat, dtype=float)
    comps = build_score_components(design, X, Y, fit.mu_models, tau_hat)
    return NuisanceFit(tau_hat=tau_hat, mu_models=fit.mu_models, comps=comps, fit_result=fit)

def estimate_static_nuisances_by_source_t(
    design: helpers.SADDesign,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    random_state: int,
    rv_mu_degree: int=2,
    rv_b_degree: int=2,
    ridge_alpha: float=RIDGE_ALPHA,
    true_mu: Optional[np.ndarray]=None,
    true_moment: Optional[np.ndarray]=None,
) -> Dict[int, NuisanceFit]:
    fit = rc.fit_rv_time_specific(
        X,
        Y,
        design,
        mu_degree=int(rv_mu_degree),
        b_degree=int(rv_b_degree),
        ridge_alpha=float(ridge_alpha),
        option='nonparametric',
        assumption='static',
        mu_gbrt_params=dict(MPDTA_MU_GBRT_PARAMS),
        b_gbrt_params=dict(MPDTA_RV_MOMENT_GBRT_PARAMS),
        random_state=random_state,
        true_mu=true_mu,
        true_b=true_moment,
    )
    out: Dict[int, NuisanceFit] = {}
    for t in range(1, design.T + 1):
        tau_hat = np.tile(np.asarray(fit.tau_hat_by_t[:, t - 1], dtype=float)[:, None], (1, design.T))
        comps = build_score_components(design, X, Y, fit.mu_models, tau_hat)
        out[t] = NuisanceFit(tau_hat=tau_hat, mu_models=fit.mu_models, comps=comps, fit_result=fit)
    return out

def lag_h0l_pvalue(
    comps,
    A: np.ndarray,
    *,
    l: int,
    rng: np.random.Generator,
    n_perms: int,
) -> Tuple[float, float]:
    stat_fn = helpers.likelihood_score_stat_tl_from_A
    perm_fn = helpers.likelihood_score_stats_tl_from_perms
    obs, perms = ([], [])
    for t in range(1, int(comps.T) - l + 1):
        P = helpers.perms_H_tl(A, t=t, l=l, T=int(comps.T), rng=rng, n_perms=n_perms)
        obs.append(stat_fn(A, comps, t=t, l=l))
        perms.append(perm_fn(P, comps, t=t, l=l))
    stat = float(np.mean(obs)) if obs else 0.0
    perm = np.column_stack(perms).mean(axis=1) if perms else np.zeros(n_perms)
    return (float(helpers.right_tailed_pvalue(stat, np.asarray(perm, dtype=float))), stat)

def global_h0_pvalue(
    comps,
    A: np.ndarray,
    *,
    rng: np.random.Generator,
    n_perms: int,
) -> Tuple[float, float]:
    P = helpers.perms_H0(np.asarray(A, dtype=int), rng, n_perms)
    stat = helpers.likelihood_score_stat_global_from_A(A, comps)
    perm = helpers.likelihood_score_stats_global_from_perms(P, comps)
    return (float(helpers.right_tailed_pvalue(float(stat), np.asarray(perm, dtype=float))), float(stat))

def _tw_demean(v: np.ndarray, n_units: int, n_periods: int) -> np.ndarray:
    m = np.asarray(v, float).reshape(n_units, n_periods)
    return (m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + m.mean()).reshape(-1)

def _cluster_twfe_fit(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    n_units, n_periods = np.asarray(Y).shape
    y = _tw_demean(np.asarray(Y, float).reshape(-1, order='C'), n_units, n_periods)
    X = np.column_stack([_tw_demean(np.asarray(X[:, j], float), n_units, n_periods) for j in range(X.shape[1])])
    keep = np.where(np.sum(X ** 2, axis=0) > 1e-12)[0]
    X = X[:, keep]
    if X.shape[1] == 0:
        return (np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float), n_units)
    XtX_inv = np.linalg.pinv(X.T @ X + 1e-08 * np.eye(X.shape[1]))
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = sum((np.outer(X[i * n_periods:(i + 1) * n_periods].T @ resid[i * n_periods:(i + 1) * n_periods],
                         X[i * n_periods:(i + 1) * n_periods].T @ resid[i * n_periods:(i + 1) * n_periods]) for i in range(n_units)))
    return (beta, n_units / max(n_units - 1, 1) * (XtX_inv @ meat @ XtX_inv), n_units)

def _twfe_static_fit(Y: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    A = np.asarray(A, dtype=int)
    n_units, n_periods = np.asarray(Y).shape
    time = np.tile(np.arange(1, n_periods + 1), n_units)
    cohort = np.repeat(A, n_periods)
    return _cluster_twfe_fit(Y, ((cohort <= n_periods) & (time >= cohort)).astype(float)[:, None])

def _event_design(A: np.ndarray, T: int) -> Tuple[np.ndarray, List[str]]:
    A = np.asarray(A, dtype=int)
    time = np.tile(np.arange(1, T + 1), A.size)
    cohort = np.repeat(A, T)
    rel, ever = (time - cohort, cohort <= T)
    cols, names = ([], [])
    for r in range(-(T - 1), 0):
        if r != -1:
            cols.append((ever & (rel == r)).astype(float))
            names.append(f'rel_{r}')
    for r in TWFE_EVENT_LAGS:
        cols.append((ever & (rel == r)).astype(float))
        names.append(f'rel_{r}')
    cols.append((ever & (rel >= max(TWFE_EVENT_LAGS) + 1)).astype(float))
    names.append(f'rel_ge_{max(TWFE_EVENT_LAGS) + 1}')
    return (np.column_stack(cols), names)

def _twfe_event_study_fit(
    Y: np.ndarray,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    n_units, n_periods = np.asarray(Y).shape
    X, names = _event_design(A, n_periods)
    X_dm = np.column_stack([_tw_demean(np.asarray(X[:, j], float), n_units, n_periods) for j in range(X.shape[1])])
    keep = np.where(np.sum(X_dm ** 2, axis=0) > 1e-12)[0]
    beta, V, n_units = _cluster_twfe_fit(Y, X)
    return (beta, V, [names[j] for j in keep], n_units)

def _single_coef_test(beta: np.ndarray, V: np.ndarray, j: int, n_units: int) -> TestResult:
    if beta.size == 0 or V.size == 0 or j < 0 or (j >= beta.size):
        return TestResult('TWFE', 1.0, 0.0)
    se = float(np.sqrt(max(V[j, j], 0.0)))
    if not np.isfinite(se) or se <= 1e-12:
        return TestResult('TWFE', 1.0, float(beta[j]))
    t = float(beta[j] / se)
    p = float(2.0 * stdtr(max(n_units - 1, 1), -abs(t)))
    return TestResult('TWFE', min(max(p, 1e-12), 1.0), float(beta[j]))

def twfe_h0l_pvalues(Y: np.ndarray, A: np.ndarray, *, l: int) -> List[TestResult]:
    beta, V, names, n_units = _twfe_event_study_fit(Y, A)
    target = f'rel_ge_{max(TWFE_EVENT_LAGS) + 1}' if l >= max(TWFE_EVENT_LAGS) + 1 else f'rel_{l}'
    if target not in names:
        return [TestResult('TWFE', 1.0, 0.0)]
    return [_single_coef_test(beta, V, names.index(target), n_units)]

def twfe_h0_pvalues(Y: np.ndarray, A: np.ndarray) -> List[TestResult]:
    beta, V, n_units = _twfe_static_fit(Y, A)
    return [_single_coef_test(beta, V, 0, n_units)]

def subgroup_threshold_median(data: MPDTAData) -> float:
    """True subgroup threshold: top 35% lpop is the negative-effect subgroup."""
    return float(np.quantile(data.X_raw[data.gating_variable].to_numpy(dtype=float), 0.65))

def default_effective_subgroup_tau(data: MPDTAData) -> float:
    return 0.5 * float(np.std(data.Y_obs))

def build_gate_tau_matrix(
    data: MPDTAData,
    *,
    threshold: float,
    effective_tau: Optional[float]=None,
    null_tau: float=0.0,
) -> np.ndarray:
    """Static subgroup DGP with a negative effect in the high-lpop subgroup.

    The command-line tau is interpreted as a positive magnitude. Units with
    gate > threshold receive -tau; units at or below the threshold are true nulls.
    """
    gate = data.X_raw[data.gating_variable].to_numpy(dtype=float)
    magnitude = default_effective_subgroup_tau(data) if effective_tau is None else abs(float(effective_tau))
    tau0 = np.where(gate > float(threshold), -magnitude, float(null_tau))
    return np.repeat(np.asarray(tau0, float)[:, None], len(data.years), axis=1)

def _subgroup_default_params(kind: NonparametricKind, min_samples_leaf: int) -> Dict[str, object]:
    kind = normalize_nonparametric_kind(kind)
    if kind == 'gbrt':
        params = dict(SUBGROUP_GBRT_DEFAULTS['mu'])
        params['min_samples_leaf'] = int(min_samples_leaf)
        return params
    return {'max_depth': 1, 'min_samples_leaf': int(min_samples_leaf)}

def _fit_subgroup_b_models(
    X: np.ndarray,
    targets: np.ndarray,
    *,
    random_state: int,
    nonparametric_kind: NonparametricKind,
    b_params: Optional[Dict[str, object]]=None,
) -> List:
    kind = normalize_nonparametric_kind(nonparametric_kind)
    params = _subgroup_default_params(kind, 1) if b_params is None else dict(b_params)
    out = []
    for j in range(targets.shape[1]):
        reg = helpers.make_regressor(
            degree=1,
            ridge_alpha=RIDGE_ALPHA,
            option='nonparametric',
            nonparametric_kind=kind,
            random_state=random_state + j,
            gbrt_params=params,
        )
        reg.fit(X, targets[:, j])
        out.append(reg)
    return out

def fit_subgroup_focal_model(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    random_state: int,
    nonparametric_kind: NonparametricKind='gbrt',
    mu_params: Optional[Dict[str, object]]=None,
    b_params: Optional[Dict[str, object]]=None,
) -> SubgroupFocalModel:
    kind = normalize_nonparametric_kind(nonparametric_kind)
    default_mu_params = _subgroup_default_params(kind, 20)
    mu_models, R = rc._fit_mu_residuals(
        X,
        Y,
        degree=1,
        ridge_alpha=RIDGE_ALPHA,
        option='nonparametric',
        assumption='static',
        random_state=random_state,
        gbrt_params=default_mu_params if mu_params is None else dict(mu_params),
        nonparametric_kind=kind,
    )
    b_models = _fit_subgroup_b_models(X, R ** 2, random_state=random_state, nonparametric_kind=kind, b_params=b_params)
    return SubgroupFocalModel(mu_models=mu_models, b_models=b_models)

def predict_subgroup_tau_hat(
    model: SubgroupFocalModel,
    X: np.ndarray,
    design: helpers.SADDesign,
) -> np.ndarray:
    b_hat = np.column_stack([m.predict(X) for m in model.b_models]) if model.b_models else np.empty((X.shape[0], 0), float)
    v = np.array([design.pi_leq[t] * (1.0 - design.pi_leq[t]) for t in range(1, design.T + 1)], float)
    centered_v = v - v.mean()
    mask = np.abs(centered_v) >= 1e-10
    if not np.any(mask):
        tau = np.zeros(X.shape[0], dtype=float)
    else:
        centered_b = b_hat[:, mask] - b_hat.mean(axis=1, keepdims=True)
        tau = np.sqrt(np.maximum(centered_b @ centered_v[mask] / float(np.sum(centered_v[mask] ** 2)), 0.0))
    return np.tile(tau[:, None], (1, design.T))

def resolve_subgroup_tau_signs(
    model: SubgroupFocalModel,
    X: np.ndarray,
    Y: np.ndarray,
    design: helpers.SADDesign,
    tau_pm: np.ndarray,
) -> np.ndarray:
    """Apply the assignment-free unit-level sign stage to a subgroup RV estimate."""
    residuals = np.asarray(Y, dtype=float) - helpers.predict_mu(model.mu_models, X)
    tau_signed, _, _, _ = helpers.resolve_unit_signs_from_residual_trajectory(
        design,
        residuals,
        np.asarray(tau_pm, dtype=float),
    )
    return tau_signed

def select_threshold_stump_from_tau_hat(
    gate: np.ndarray,
    tau_hat: np.ndarray,
    *,
    min_frac: float=0.1,
) -> Dict[str, float]:
    gate = np.asarray(gate, dtype=float)
    signal = np.asarray(tau_hat, dtype=float)
    signal = signal if signal.ndim == 1 else signal[:, 0]
    n = int(gate.shape[0])
    if n == 0:
        return {
            'threshold': 0.0,
            'threshold_left': 0.0,
            'threshold_right': 0.0,
            'left_mean_tau_hat': 0.0,
            'right_mean_tau_hat': 0.0,
            'stump_sse': 0.0,
            'stump_mse': 0.0,
            'stump_gain': 0.0,
            'stump_gap': 0.0,
            'min_leaf': 0,
            'split_index': -1,
        }
    order = np.argsort(gate)
    x = gate[order]
    y = signal[order]
    y2 = y ** 2
    min_leaf = max(5, int(np.ceil(float(min_frac) * n)))
    if n < 2 * min_leaf:
        mean_tau = float(np.mean(y))
        threshold = float(np.median(x))
        stump_sse = float(np.sum((y - mean_tau) ** 2))
        return {
            'threshold': threshold,
            'threshold_left': threshold,
            'threshold_right': threshold,
            'left_mean_tau_hat': mean_tau,
            'right_mean_tau_hat': mean_tau,
            'stump_sse': stump_sse,
            'stump_mse': stump_sse / max(n, 1),
            'stump_gain': 0.0,
            'stump_gap': 0.0,
            'min_leaf': int(min_leaf),
            'split_index': -1,
        }
    csum = np.cumsum(y)
    csum2 = np.cumsum(y2)
    total = float(csum[-1])
    total2 = float(csum2[-1])
    sse_const = float(total2 - total * total / n)
    candidates = []
    for k in range(min_leaf - 1, n - min_leaf):
        nl = k + 1
        nr = n - nl
        left_sum = float(csum[k])
        left_sum2 = float(csum2[k])
        right_sum = total - left_sum
        right_sum2 = total2 - left_sum2
        left_mean = left_sum / nl
        right_mean = right_sum / nr
        stump_sse = float(left_sum2 - left_sum * left_sum / nl + right_sum2 - right_sum * right_sum / nr)
        candidates.append({
            'k': int(k),
            'sse': stump_sse,
            'mse': stump_sse / n,
            'gain': sse_const - stump_sse,
            'gap': right_mean - left_mean,
            'left_mean': left_mean,
            'right_mean': right_mean,
        })
    # Current subgroup DGP has a high-lpop effective subgroup with negative effects,
    # so the right/high side should have a lower signed tau mean than the left/low side.
    feasible = [c for c in candidates if c['right_mean'] <= c['left_mean']]
    chosen = min(feasible or candidates, key=lambda c: (float(c['mse']), float(c['gap']), int(c['k'])))
    k = int(chosen['k'])
    threshold_left = float(x[k])
    threshold_right = float(x[k + 1])
    return {
        'threshold': float(0.5 * (threshold_left + threshold_right)),
        'threshold_left': threshold_left,
        'threshold_right': threshold_right,
        'left_mean_tau_hat': float(chosen['left_mean']),
        'right_mean_tau_hat': float(chosen['right_mean']),
        'stump_sse': float(chosen['sse']),
        'stump_mse': float(chosen['mse']),
        'stump_gain': float(chosen['gain']),
        'stump_gap': float(chosen['gap']),
        'min_leaf': int(min_leaf),
        'split_index': k,
    }

def _oriented_subgroup_pvalue(
    stat: float,
    perm: np.ndarray,
    comps,
    idx: np.ndarray,
    *,
    statistic: Statistic,
) -> Tuple[float, float, float]:
    """Return p-value, reported statistic, and orientation for subgroup tests.

    AIPW estimates a signed average effect.  Under the current subgroup DGP the
    effective subgroup has negative effects, so a negative statistic should be
    evidence against the null.  We orient the statistic by the estimated average
    signed CATE in the selected subgroup; this orientation is fixed conditional
    on the fitted nuisances and is applied identically to observed/permuted
    assignments.  Likelihood scores are already aligned with tau_hat and remain
    right-tailed without extra orientation.
    """
    stat = float(stat)
    perm = np.asarray(perm, dtype=float)
    if normalize_statistic(statistic) == 'aipw':
        tau_sub = np.asarray(getattr(comps, 'tau_hat'), dtype=float)[np.asarray(idx, dtype=int)]
        avg_tau = float(np.mean(tau_sub)) if tau_sub.size else 0.0
        orient = -1.0 if avg_tau < 0.0 else 1.0
        return (float(helpers.right_tailed_pvalue(orient * stat, orient * perm)), float(stat), float(orient))
    return (float(helpers.right_tailed_pvalue(stat, perm)), float(stat), 1.0)

def subgroup_pvalue(
    comps,
    A: np.ndarray,
    subgroup_idx: np.ndarray,
    *,
    statistic: Statistic='likelihood',
    rng: np.random.Generator,
    n_perms: int,
) -> TestResult:
    statistic = normalize_statistic(statistic)
    idx = np.asarray(subgroup_idx, dtype=int)
    if idx.size == 0:
        return TestResult('RV', 1.0, 0.0)
    P = helpers.perms_H_0k(A, idx, rng, n_perms)
    if statistic == 'aipw':
        stat = helpers.aipw_score_stat_global_from_A(A, comps, subset=idx)
        perm = helpers.aipw_score_stats_global_from_perms(P, comps, subset=idx)
    else:
        stat = helpers.likelihood_score_stat_global_from_A(A, comps, subset=idx)
        perm = helpers.likelihood_score_stats_global_from_perms(P, comps, subset=idx)
    pvalue, reported_stat, _ = _oriented_subgroup_pvalue(
        float(stat),
        np.asarray(perm, dtype=float),
        comps,
        idx,
        statistic=statistic,
    )
    return TestResult('RV', pvalue, reported_stat)

def summarize_rejections(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    return df.groupby(list(group_cols), as_index=False).agg(reject_rate=('reject', 'mean'),
                                                            mean_pvalue=('pvalue', 'mean'),
                                                            mean_stat=('stat', 'mean'),
                                                            n=('reject', 'size')).sort_values(list(group_cols)).reset_index(drop=True)
