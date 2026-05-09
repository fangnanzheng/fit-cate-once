from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import helpers
import numpy as np

import rc
DEFAULT_COHORT_PROPS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
DEFAULT_N_GRID = [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
DEFAULT_M_GRID = [25, 50, 100, 200, 400, 800, 1600, 3200]
DEFAULT_NOISE_SD = 0.1
DEFAULT_LAGGED_DGP = 'shared_easy'
DEFAULT_START_N = 25600
DEFAULT_START_SIGN_THRESHOLD = 0.0
DEFAULT_START_STAGE2_MAX_DEPTH = 1
DEFAULT_START_STAGE2_MIN_LEAF = 20
DEFAULT_START_STAGE2_N_ESTIMATORS = 60
DEFAULT_START_STAGE2_LEARNING_RATE = 0.05
DEFAULT_START_EPS = 1e-08
_FAST_PLATEAU_NONPARAMETRIC_MU_CFG: Dict[str, Any] = {
    'n_estimators': 40,
    'learning_rate': 0.1,
    'max_depth': 1,
    'subsample': 1.0,
    'min_samples_leaf': 50,
}
_FAST_PLATEAU_NONPARAMETRIC_MOMENT_CFG: Dict[str, Any] = {
    'n_estimators': 60,
    'learning_rate': 0.1,
    'max_depth': 1,
    'subsample': 1.0,
    'min_samples_leaf': 50,
}
STATIC_NONPARAMETRIC_MU_CONFIG_MAP: Dict[int, Dict[str, Any]] = {N: dict(_FAST_PLATEAU_NONPARAMETRIC_MU_CFG) for N in DEFAULT_N_GRID}
STATIC_NONPARAMETRIC_MOMENT_CONFIG_MAP: Dict[int, Dict[str, Any]] = {N: dict(_FAST_PLATEAU_NONPARAMETRIC_MOMENT_CFG) for N in DEFAULT_N_GRID}
LAGGED_NONPARAMETRIC_MU_CONFIG_MAP: Dict[int, Dict[str, Any]] = {N: dict(_FAST_PLATEAU_NONPARAMETRIC_MU_CFG) for N in DEFAULT_N_GRID + 
                                                                 [DEFAULT_START_N]}
LAGGED_NONPARAMETRIC_MOMENT_CONFIG_MAP: Dict[int, Dict[str, Any]] = {N: dict(_FAST_PLATEAU_NONPARAMETRIC_MOMENT_CFG) for N in DEFAULT_N_GRID + 
                                                                     [DEFAULT_START_N]}

@dataclass
class GeneratedPanel:
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    mu0_true: np.ndarray
    tau_true: np.ndarray
    tau_mag_true: np.ndarray
    sign_true: np.ndarray
    mu_true: np.ndarray
    design: helpers.SADDesign
    dgp_name: str

@dataclass
class Metrics:
    rmse: float
    nmse: float
    mse: float
    variance: float

    def as_dict(self, prefix: str='') -> Dict[str, float]:
        return {
            f'{prefix}rmse': self.rmse,
            f'{prefix}nmse': self.nmse,
            f'{prefix}mse': self.mse,
            f'{prefix}variance': self.variance,
        }

@dataclass
class StaticStageResult:
    mu_models: List[Any]
    mu_hat: np.ndarray
    diag_hat: np.ndarray
    tau_hat: np.ndarray

def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in str(spec).split(',') if x.strip()]

def parse_float_list(spec: str) -> List[float]:
    return [float(x.strip()) for x in str(spec).split(',') if x.strip()]

def counts_for_sample_size(N: int, cohort_props: Sequence[float]=DEFAULT_COHORT_PROPS) -> List[int]:
    props = np.asarray(cohort_props, dtype=float)
    if len(props) == 0:
        raise ValueError('cohort_props must be non-empty')
    if np.any(props < 0):
        raise ValueError('cohort_props must be nonnegative')
    if not np.isclose(props.sum(), 1.0):
        raise ValueError(f'cohort_props must sum to 1; got {props.sum()}')
    raw = N * props
    counts = np.floor(raw).astype(int)
    remainder = int(N - counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        counts[order[:remainder]] += 1
    elif remainder < 0:
        frac = raw - counts
        order = np.argsort(frac)
        counts[order[:-remainder]] -= 1
    return counts.tolist()

def build_design_for_N(
    N: int,
    T: int=5,
    cohort_props: Sequence[float]=DEFAULT_COHORT_PROPS,
) -> helpers.SADDesign:
    counts = counts_for_sample_size(N, cohort_props=cohort_props)
    if len(counts) != T + 1:
        raise ValueError(f'cohort_props must have length T+1={T + 1}; got {len(counts)}')
    return helpers.SADDesign(T=T, counts_by_time=counts)

def sample_one_dim_covariates(N: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(-1.0, 1.0, size=N)
    return x[:, None]

def untreated_mean_matrix_linear(X: np.ndarray, T: int) -> np.ndarray:
    x = np.asarray(X[:, 0], dtype=float)
    intercepts = 0.55 + 0.1 * np.arange(T, dtype=float)
    slopes = 0.3 - 0.03 * np.arange(T, dtype=float)
    return intercepts[None, :] + x[:, None] * slopes[None, :]

def static_tau_matrix_simple(X: np.ndarray, T: int) -> np.ndarray:
    x = np.asarray(X[:, 0], dtype=float)
    tau_star = np.sqrt(1.15 + 0.2 * x)
    return np.repeat(tau_star[:, None], T, axis=1)

def lagged_tau_matrix_shared_easy(X: np.ndarray, T: int) -> np.ndarray:
    x = np.asarray(X[:, 0], dtype=float)
    base = np.linspace(1.3, 0.7, T)
    slope = np.zeros(T, dtype=float)
    slope[-1] = 0.35
    return base[None, :] + x[:, None] * slope[None, :]

def start_sign_function(
    X: np.ndarray,
    *,
    threshold: float=DEFAULT_START_SIGN_THRESHOLD,
) -> np.ndarray:
    x = np.asarray(X[:, 0], dtype=float)
    return np.where(x >= float(threshold), 1.0, -1.0)

def generate_errors(N: int, T: int, *, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=float(noise_sd), size=(N, T))

def true_observed_mean(
    mu0_true: np.ndarray,
    tau_true: np.ndarray,
    design: helpers.SADDesign,
) -> np.ndarray:
    T = design.T
    mu = np.array(mu0_true, dtype=float, copy=True)
    for u in range(1, T + 1):
        weights = design.pi[1:u + 1][::-1]
        mu[:, u - 1] += np.sum(weights[None, :] * tau_true[:, :u], axis=1)
    return mu

def generate_panel(
    N: int,
    *,
    assumption: str,
    option: str,
    seed: int,
    T: int=5,
    cohort_props: Sequence[float]=DEFAULT_COHORT_PROPS,
    noise_sd: float=DEFAULT_NOISE_SD,
    experiment: str='raw',
    sign_threshold: float=DEFAULT_START_SIGN_THRESHOLD,
) -> GeneratedPanel:
    rng = np.random.default_rng(seed)
    experiment = str(experiment).strip().lower()
    design = build_design_for_N(N, T=T, cohort_props=cohort_props)
    X = sample_one_dim_covariates(N, rng)
    A = design.sample_A(rng)
    mu0_true = untreated_mean_matrix_linear(X, T)
    global_sign = 1.0
    tau_true, tau_mag_true, sign_true = tau_matrix_for_experiment(
        X,
        assumption=assumption,
        T=T,
        experiment=experiment,
        sign_threshold=sign_threshold,
        global_sign=global_sign,
    )
    eps = generate_errors(N, T, noise_sd=noise_sd, rng=rng)
    Y = mu0_true + eps
    for i, a in enumerate(A):
        if a <= T:
            Y[i, a - 1:] += tau_true[i, :T - a + 1]
    mu_true = true_observed_mean(mu0_true, tau_true, design)
    return GeneratedPanel(
        X=X,
        A=A,
        Y=Y,
        mu0_true=mu0_true,
        tau_true=tau_true,
        tau_mag_true=tau_mag_true,
        sign_true=sign_true,
        mu_true=mu_true,
        design=design,
        dgp_name=dgp_name(assumption, option, experiment),
    )

def flatten_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    var_from: Optional[np.ndarray]=None,
) -> Metrics:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f'shape mismatch: {pred.shape} vs {target.shape}')
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    variance_source = target if var_from is None else np.asarray(var_from, dtype=float)
    variance = float(np.var(variance_source))
    nmse = float(mse / variance) if variance > 0 else float('nan')
    return Metrics(rmse=float(np.sqrt(mse)), nmse=nmse, mse=mse, variance=variance)

def tau_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    sign_invariant: bool,
) -> Metrics:
    """Compute tau NMSE as E[d(pred, target)^2] / E[||target||_2^2].

    For assignment-free CATE estimates identified only up to sign,
    d is the per-unit +/- distance min(||pred-target||, ||pred+target||).
    For signed estimators, d is the ordinary Euclidean distance.  The Metrics
    ``variance`` field is kept for output-schema compatibility and stores the
    denominator E[||target||_2^2], not a centered variance.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f'shape mismatch: {pred.shape} vs {target.shape}')
    if pred.ndim == 1:
        pred = pred[:, None]
        target = target[:, None]
    if pred.ndim != 2:
        raise ValueError(f'tau metrics expect a vector or matrix; got ndim={pred.ndim}')
    sq_err = np.sum((pred - target) ** 2, axis=1)
    if sign_invariant:
        sq_err_alt = np.sum((pred + target) ** 2, axis=1)
        sq_err = np.minimum(sq_err, sq_err_alt)
    mse = float(np.mean(sq_err))
    denom = float(np.mean(np.sum(target ** 2, axis=1)))
    nmse = float(mse / denom) if denom > 0 else float('nan')
    return Metrics(rmse=float(np.sqrt(mse)), nmse=nmse, mse=mse, variance=denom)

def build_offdiag_pairs(T: int) -> List[Tuple[int, int]]:
    return [(t, s) for t in range(2, T + 1) for s in range(1, t)]

def build_Htilde_stack(design: helpers.SADDesign) -> np.ndarray:
    T = design.T
    pi = design.pi

    def _J(k: int) -> np.ndarray:
        J = np.zeros((T, T), dtype=float)
        for a in range(T):
            b = a - k
            if 0 <= b < T:
                J[a, b] = 1.0
        return J

    def _pi_trunc(t: int) -> np.ndarray:
        v = np.zeros(T, dtype=float)
        for j in range(1, t + 1):
            v[t - j] = float(pi[j])
        return v

    def _D(s: int) -> np.ndarray:
        d = np.zeros(T, dtype=float)
        for j in range(1, s + 1):
            d[s - j] = float(pi[j])
        return np.diag(d)
    mats = []
    for t, s in build_offdiag_pairs(T):
        H = _J(t - s) @ _D(s) - np.outer(_pi_trunc(t), _pi_trunc(s))
        mats.append(0.5 * (H + H.T))
    return np.stack(mats, axis=0)

def oracle_diag_moments(
    tau_true: np.ndarray,
    design: helpers.SADDesign,
    *,
    noise_sd: float=DEFAULT_NOISE_SD,
) -> np.ndarray:
    v = np.array([design.pi_leq[t] * (1.0 - design.pi_leq[t]) for t in range(1, design.T + 1)], dtype=float)
    tau_star = tau_true[:, [0]]
    sigma2 = float(noise_sd) ** 2
    return sigma2 + tau_star ** 2 * v[None, :]

def oracle_offdiag_moments(tau_true: np.ndarray, design: helpers.SADDesign) -> np.ndarray:
    Htilde = build_Htilde_stack(design)
    return np.einsum('na,jab,nb->nj', tau_true, Htilde, tau_true)

def observed_offdiag_products(R: np.ndarray) -> np.ndarray:
    N, T = R.shape
    out = np.zeros((N, T * (T - 1) // 2), dtype=float)
    for j, (t, s) in enumerate(build_offdiag_pairs(T)):
        out[:, j] = R[:, t - 1] * R[:, s - 1]
    return out

def fit_multi_target_regression(
    X: np.ndarray,
    targets: np.ndarray,
    *,
    option: str,
    degree: int,
    ridge_alpha: float,
    random_state: Optional[int],
    gbrt_params: Optional[Dict[str, Any]]=None,
) -> np.ndarray:
    out = np.zeros_like(targets, dtype=float)
    for j in range(targets.shape[1]):
        rs = None if random_state is None else int(random_state) + j
        reg = helpers.make_regressor(
            degree=degree,
            ridge_alpha=ridge_alpha,
            option=option,
            random_state=rs,
            gbrt_params=gbrt_params,
        )
        reg.fit(X, targets[:, j])
        out[:, j] = reg.predict(X)
    return out

def fit_static_stagewise(
    X: np.ndarray,
    Y: np.ndarray,
    design: helpers.SADDesign,
    *,
    option: str,
    mu_degree: int=1,
    b_degree: int=1,
    ridge_alpha: float=0.0,
    mu_gbrt_params: Optional[Dict[str, Any]]=None,
    b_gbrt_params: Optional[Dict[str, Any]]=None,
    random_state: Optional[int]=None,
    denom_eps: float=1e-10,
) -> StaticStageResult:
    mu_models = helpers.fit_mu_models(
        X,
        Y,
        degree=mu_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        random_state=random_state,
        gbrt_params=mu_gbrt_params,
    )
    mu_hat = helpers.predict_mu(mu_models, X)
    R = Y - mu_hat
    diag_hat = fit_multi_target_regression(
        X,
        R ** 2,
        option=option,
        degree=b_degree,
        ridge_alpha=ridge_alpha,
        random_state=None if random_state is None else int(random_state) + 10000,
        gbrt_params=b_gbrt_params,
    )
    b_bar = diag_hat.mean(axis=1)
    v = np.array([design.pi_leq[t] * (1.0 - design.pi_leq[t]) for t in range(1, design.T + 1)], dtype=float)
    x = v - v.mean()
    mask = np.abs(x) >= denom_eps
    tau_star = np.zeros(X.shape[0], dtype=float)
    if np.any(mask):
        x_m = x[mask]
        denom = float(np.sum(x_m ** 2))
        y = diag_hat[:, mask] - b_bar[:, None]
        tau2 = y @ x_m / denom
        tau_star = np.sqrt(np.maximum(tau2, 0.0))
    tau_hat = np.tile(tau_star[:, None], (1, design.T))
    return StaticStageResult(mu_models=mu_models, mu_hat=mu_hat, diag_hat=diag_hat, tau_hat=tau_hat)

def fit_tau_estimator(
    panel: GeneratedPanel,
    *,
    assumption: str,
    option: str,
    mu_degree: int=1,
    b_degree: int=1,
    basis_degree: int=1,
    ridge_alpha: float=0.0,
    mu_gbrt_params: Optional[Dict[str, Any]]=None,
    moment_gbrt_params: Optional[Dict[str, Any]]=None,
    random_state: Optional[int]=None,
    n_starts: int=5,
    max_nfev: int=4000,
    convex_max_iters: int=2000,
    lambda_trace: float=0.0,
    nls_ridge: float=0.0,
    refine_nls: bool=True,
    use_convex_init: bool=True,
) -> np.ndarray:
    if assumption == 'static':
        res = fit_static_stagewise(
            panel.X,
            panel.Y,
            panel.design,
            option=option,
            mu_degree=mu_degree,
            b_degree=b_degree,
            ridge_alpha=ridge_alpha,
            mu_gbrt_params=mu_gbrt_params,
            b_gbrt_params=moment_gbrt_params,
            random_state=random_state,
        )
        return res.tau_hat
    res = rc.fit_rc(
        panel.X,
        panel.Y,
        panel.design,
        mu_degree=mu_degree,
        basis_degree=basis_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        mu_gbrt_params=mu_gbrt_params,
        moment_denoise=True,
        moment_gbrt_params=moment_gbrt_params,
        n_starts=n_starts,
        max_nfev=max_nfev,
        random_state=random_state,
        lambda_trace=lambda_trace,
        convex_max_iters=convex_max_iters,
        use_convex_init=use_convex_init,
        refine_nls=refine_nls,
        nls_ridge=nls_ridge,
    )
    return np.asarray(res.tau_hat, dtype=float)

def evaluate_mu_once(
    panel: GeneratedPanel,
    *,
    option: str,
    mu_degree: int,
    ridge_alpha: float,
    mu_gbrt_params: Optional[Dict[str, Any]],
    random_state: Optional[int],
) -> Metrics:
    mu_models = helpers.fit_mu_models(
        panel.X,
        panel.Y,
        degree=mu_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        random_state=random_state,
        gbrt_params=mu_gbrt_params,
    )
    mu_hat = helpers.predict_mu(mu_models, panel.X)
    return flatten_metrics(mu_hat, panel.mu_true)

def evaluate_moment_once(
    panel: GeneratedPanel,
    *,
    assumption: str,
    option: str,
    mu_degree: int,
    moment_degree: int,
    ridge_alpha: float,
    mu_gbrt_params: Optional[Dict[str, Any]],
    moment_gbrt_params: Optional[Dict[str, Any]],
    random_state: Optional[int]=None,
    noise_sd: float=DEFAULT_NOISE_SD,
) -> Metrics:
    mu_models = helpers.fit_mu_models(
        panel.X,
        panel.Y,
        degree=mu_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        random_state=random_state,
        gbrt_params=mu_gbrt_params,
    )
    R = panel.Y - helpers.predict_mu(mu_models, panel.X)
    if assumption == 'static':
        moment_hat = fit_multi_target_regression(
            panel.X,
            R ** 2,
            option=option,
            degree=moment_degree,
            ridge_alpha=ridge_alpha,
            random_state=None if random_state is None else int(random_state) + 10000,
            gbrt_params=moment_gbrt_params,
        )
        target = oracle_diag_moments(panel.tau_true, panel.design, noise_sd=noise_sd)
    else:
        moment_hat = fit_multi_target_regression(
            panel.X,
            observed_offdiag_products(R),
            option=option,
            degree=moment_degree,
            ridge_alpha=ridge_alpha,
            random_state=None if random_state is None else int(random_state) + 10000,
            gbrt_params=moment_gbrt_params,
        )
        target = oracle_offdiag_moments(panel.tau_true, panel.design)
    return flatten_metrics(moment_hat, target)

def evaluate_tau_once(
    panel: GeneratedPanel,
    *,
    assumption: str,
    option: str,
    mu_degree: int,
    moment_degree: int,
    basis_degree: int,
    ridge_alpha: float,
    mu_gbrt_params: Optional[Dict[str, Any]],
    moment_gbrt_params: Optional[Dict[str, Any]],
    random_state: Optional[int],
    n_starts: int,
    max_nfev: int,
    convex_max_iters: int,
    lambda_trace: float,
    nls_ridge: float,
    refine_nls: bool,
    use_convex_init: bool,
) -> Metrics:
    tau_hat = fit_tau_estimator(
        panel,
        assumption=assumption,
        option=option,
        mu_degree=mu_degree,
        b_degree=moment_degree,
        basis_degree=basis_degree,
        ridge_alpha=ridge_alpha,
        mu_gbrt_params=mu_gbrt_params,
        moment_gbrt_params=moment_gbrt_params,
        random_state=random_state,
        n_starts=n_starts,
        max_nfev=max_nfev,
        convex_max_iters=convex_max_iters,
        lambda_trace=lambda_trace,
        nls_ridge=nls_ridge,
        refine_nls=refine_nls,
        use_convex_init=use_convex_init,
    )
    return tau_metrics(tau_hat, panel.tau_true, sign_invariant=True)

def _weighted_poly_ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_pred: np.ndarray,
    *,
    degree: int,
    ridge_alpha: float,
) -> np.ndarray:
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    X_pred = np.asarray(X_pred, dtype=float)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Phi_train = poly.fit_transform(X_train)
    Phi_pred = poly.transform(X_pred)
    scaler = StandardScaler()
    Phi_train_s = scaler.fit_transform(Phi_train)
    Phi_pred_s = scaler.transform(Phi_pred)
    w = np.maximum(sample_weight, 0.0)
    if not np.any(w > 0):
        return np.zeros(X_pred.shape[0], dtype=float)
    sw = np.sqrt(w)[:, None]
    Xw = Phi_train_s * sw
    yw = y_train * sw[:, 0]
    p = Xw.shape[1]
    beta = np.linalg.solve(Xw.T @ Xw + float(ridge_alpha) * np.eye(p), Xw.T @ yw)
    intercept = float(np.average(y_train - Phi_train_s @ beta, weights=w))
    return intercept + Phi_pred_s @ beta

def _eligible_mask_for_t_l(A: np.ndarray, *, t: int, lag: int) -> np.ndarray:
    u = t + lag
    return (A == t) | (A > u)

def treatment_propensity_t_l(design: helpers.SADDesign, *, t: int, lag: int) -> float:
    u = t + lag
    numer = float(design.pi[t])
    denom = float(design.pi[t] + np.sum(design.pi[u + 1:design.T + 2]))
    if denom <= 0.0:
        raise ValueError(f'Degenerate eligibility probability for t={t}, lag={lag}')
    return numer / denom

def save_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
DEFAULT_M_GRID = [25, 50, 100, 200, 400, 800, 1600, 3200]
DEFAULT_START_N = 25600
DEFAULT_START_SIGN_THRESHOLD = 0.0
DEFAULT_START_STAGE2_MAX_DEPTH = 1
DEFAULT_START_STAGE2_MIN_LEAF = 20
DEFAULT_START_STAGE2_N_ESTIMATORS = 60
DEFAULT_START_STAGE2_LEARNING_RATE = 0.05
DEFAULT_START_EPS = 1e-08
_START_NONPARAMETRIC_MU_CFG: Dict[str, Any] = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 2,
    'subsample': 1.0,
    'min_samples_leaf': 20,
}
_START_NONPARAMETRIC_MOMENT_CFG: Dict[str, Any] = {
    'n_estimators': 250,
    'learning_rate': 0.05,
    'max_depth': 2,
    'subsample': 1.0,
    'min_samples_leaf': 20,
}
_START_CONFIG_KEYS = sorted(set(DEFAULT_N_GRID + [DEFAULT_START_N]))
START_LAGGED_NONPARAMETRIC_MU_CONFIG_MAP: Dict[int, Dict[str, Any]] = {N: dict(_START_NONPARAMETRIC_MU_CFG) for N in _START_CONFIG_KEYS}
START_LAGGED_NONPARAMETRIC_MOMENT_CONFIG_MAP: Dict[int, Dict[str, Any]] = {N: dict(_START_NONPARAMETRIC_MOMENT_CFG) for N in _START_CONFIG_KEYS}

def dgp_name(assumption: str, option: str, experiment: str='raw') -> str:
    experiment = str(experiment).strip().lower()
    if experiment not in {'raw', 'start'}:
        raise ValueError("experiment must be 'raw' or 'start'")
    if assumption == 'static':
        return 'static_affine_moments'
    if experiment == 'raw':
        return 'lagged_shared_easy_linear_last_lag'
    return 'lagged_shared_easy_x_sign_warm_start'

def tau_matrix_for_experiment(
    X: np.ndarray,
    *,
    assumption: str,
    T: int,
    experiment: str='raw',
    sign_threshold: float=DEFAULT_START_SIGN_THRESHOLD,
    global_sign: float=1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    experiment = str(experiment).strip().lower()
    if assumption == 'static':
        tau = static_tau_matrix_simple(X, T)
        return (tau, np.abs(tau), np.ones(X.shape[0], dtype=float))
    tau_mag = lagged_tau_matrix_shared_easy(X, T)
    if experiment == 'raw':
        sign = np.ones(X.shape[0], dtype=float)
    elif experiment == 'start':
        _ = global_sign
        sign = start_sign_function(X, threshold=sign_threshold)
    else:
        raise ValueError("experiment must be 'raw' or 'start'")
    tau = sign[:, None] * tau_mag
    return (tau, tau_mag, sign)

def config_for_N(config_map: Mapping[int, Mapping[str, Any]], N: int) -> Dict[str, Any]:
    if N in config_map:
        return dict(config_map[N])
    if not config_map:
        return {}
    keys = sorted((int(k) for k in config_map.keys()))
    nearest = min(keys, key=lambda k: (abs(k - int(N)), k))
    return dict(config_map[nearest])

def built_in_nonparametric_config_maps(
    assumption: str,
    experiment: str='raw',
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    experiment = str(experiment).strip().lower()
    if assumption == 'static':
        return ({int(N): dict(cfg) for N, cfg in STATIC_NONPARAMETRIC_MU_CONFIG_MAP.items()},
                {int(N): dict(cfg) for N, cfg in STATIC_NONPARAMETRIC_MOMENT_CONFIG_MAP.items()})
    if experiment == 'start':
        return ({int(N): dict(cfg) for N, cfg in START_LAGGED_NONPARAMETRIC_MU_CONFIG_MAP.items()},
                {int(N): dict(cfg) for N, cfg in START_LAGGED_NONPARAMETRIC_MOMENT_CONFIG_MAP.items()})
    return ({int(N): dict(cfg) for N, cfg in LAGGED_NONPARAMETRIC_MU_CONFIG_MAP.items()},
            {int(N): dict(cfg) for N, cfg in LAGGED_NONPARAMETRIC_MOMENT_CONFIG_MAP.items()})

def _fit_weighted_stage2_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_pred: np.ndarray,
    *,
    option: str,
    degree: int,
    ridge_alpha: float,
    tree_max_depth: int,
    tree_min_leaf: int,
    random_state: Optional[int],
    n_estimators: int=DEFAULT_START_STAGE2_N_ESTIMATORS,
    learning_rate: float=DEFAULT_START_STAGE2_LEARNING_RATE,
) -> np.ndarray:
    if X_train.shape[0] == 0:
        return np.zeros(X_pred.shape[0], dtype=float)
    if option == 'parametric':
        return _weighted_poly_ridge_predict(
            X_train,
            y_train,
            sample_weight,
            X_pred,
            degree=degree,
            ridge_alpha=ridge_alpha,
        )
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(tree_max_depth),
        min_samples_leaf=int(tree_min_leaf),
        subsample=1.0,
        random_state=random_state,
    )
    reg.fit(X_train, y_train, sample_weight=sample_weight)
    return reg.predict(X_pred)

def build_rlearner_overall_dataset(
    panel: GeneratedPanel,
    mu_hat: np.ndarray,
    subset_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = panel.design.T
    subset_idx = np.asarray(subset_idx, dtype=int)
    rows_X: List[np.ndarray] = []
    rows_r: List[np.ndarray] = []
    rows_zc: List[np.ndarray] = []
    rows_i: List[np.ndarray] = []
    rows_l: List[np.ndarray] = []
    for lag in range(T):
        for t in range(1, T - lag + 1):
            eligible = _eligible_mask_for_t_l(panel.A, t=t, lag=lag)
            if subset_idx.size:
                elig_idx = subset_idx[eligible[subset_idx]]
            else:
                elig_idx = np.empty(0, dtype=int)
            if elig_idx.size == 0:
                continue
            e_t = treatment_propensity_t_l(panel.design, t=t, lag=lag)
            u = t + lag
            z = (panel.A[elig_idx] == t).astype(float)
            rows_X.append(panel.X[elig_idx])
            rows_r.append(panel.Y[elig_idx, u - 1] - mu_hat[elig_idx, u - 1])
            rows_zc.append(z - e_t)
            rows_i.append(elig_idx)
            rows_l.append(np.full(elig_idx.size, lag, dtype=int))
    if not rows_X:
        p = panel.X.shape[1]
        return (np.empty((0, p), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=int), np.empty(0, dtype=int))
    return (np.vstack(rows_X), np.concatenate(rows_r), np.concatenate(rows_zc), np.concatenate(rows_i), np.concatenate(rows_l))

def fit_warm_start_and_direct_for_overall_tau(
    panel: GeneratedPanel,
    *,
    option: str,
    M: int,
    subset_order: np.ndarray,
    mu_degree: int,
    basis_degree: int,
    ridge_alpha: float,
    mu_gbrt_params: Optional[Dict[str, Any]],
    moment_gbrt_params: Optional[Dict[str, Any]],
    random_state: Optional[int],
    n_starts: int,
    max_nfev: int,
    convex_max_iters: int,
    lambda_trace: float,
    nls_ridge: float,
    refine_nls: bool,
    use_convex_init: bool,
    stage2_degree: int,
    stage2_tree_max_depth: int,
    stage2_tree_min_leaf: int,
    stage2_n_estimators: int=DEFAULT_START_STAGE2_N_ESTIMATORS,
    stage2_learning_rate: float=DEFAULT_START_STAGE2_LEARNING_RATE,
    f_hat_precomputed: Optional[np.ndarray]=None,
    mu_hat_precomputed: Optional[np.ndarray]=None,
    stage2_eps: float=DEFAULT_START_EPS,
) -> Dict[str, Metrics]:
    N = panel.X.shape[0]
    T = panel.design.T
    M = int(max(0, min(M, N)))
    target = np.asarray(panel.tau_true, dtype=float)
    if f_hat_precomputed is None:
        tau_ws_full = fit_tau_estimator(
            panel,
            assumption='lagged',
            option=option,
            mu_degree=mu_degree,
            basis_degree=basis_degree,
            ridge_alpha=ridge_alpha,
            mu_gbrt_params=mu_gbrt_params,
            moment_gbrt_params=moment_gbrt_params,
            random_state=random_state,
            n_starts=n_starts,
            max_nfev=max_nfev,
            convex_max_iters=convex_max_iters,
            lambda_trace=lambda_trace,
            nls_ridge=nls_ridge,
            refine_nls=refine_nls,
            use_convex_init=use_convex_init,
        )
        f_hat = np.abs(np.asarray(tau_ws_full, dtype=float))
    else:
        f_hat = np.asarray(f_hat_precomputed, dtype=float)
    if mu_hat_precomputed is None:
        mu_models = helpers.fit_mu_models(
            panel.X,
            panel.Y,
            degree=mu_degree,
            ridge_alpha=ridge_alpha,
            option=option,
            random_state=None if random_state is None else int(random_state) + 50000,
            gbrt_params=mu_gbrt_params,
        )
        mu_hat = helpers.predict_mu(mu_models, panel.X)
    else:
        mu_hat = np.asarray(mu_hat_precomputed, dtype=float)
    if M == 0:
        warm_pred = f_hat.copy()
        direct_pred = np.zeros_like(target)
        return {
            'warm_start': tau_metrics(warm_pred, target, sign_invariant=True),
            'direct_from_scratch': tau_metrics(direct_pred, target, sign_invariant=False),
        }
    subset_idx = np.asarray(subset_order[:M], dtype=int)
    X_obs, y_res, z_centered, obs_unit_idx, lag_idx = build_rlearner_overall_dataset(panel, mu_hat, subset_idx)
    if X_obs.shape[0] == 0:
        warm_pred = f_hat.copy()
        direct_pred = np.zeros_like(target)
        return {
            'warm_start': tau_metrics(warm_pred, target, sign_invariant=True),
            'direct_from_scratch': tau_metrics(direct_pred, target, sign_invariant=False),
        }
    q_warm = z_centered * f_hat[obs_unit_idx, lag_idx]
    keep_warm = np.abs(q_warm) > float(stage2_eps)
    if np.any(keep_warm):
        y_tilde_warm = y_res[keep_warm] / q_warm[keep_warm]
        w_warm = q_warm[keep_warm] ** 2
        g_pred = _fit_weighted_stage2_predictor(
            X_obs[keep_warm],
            y_tilde_warm,
            w_warm,
            panel.X,
            option=option,
            degree=stage2_degree,
            ridge_alpha=ridge_alpha,
            tree_max_depth=stage2_tree_max_depth,
            tree_min_leaf=stage2_tree_min_leaf,
            n_estimators=stage2_n_estimators,
            learning_rate=stage2_learning_rate,
            random_state=None if random_state is None else int(random_state) + 100000 + M,
        )
        warm_sign = np.where(g_pred >= 0.0, 1.0, -1.0)
    else:
        warm_sign = np.ones(N, dtype=float)
    warm_pred = f_hat * warm_sign[:, None]
    direct_pred = np.zeros((N, T), dtype=float)
    for lag in range(T):
        mask = lag_idx == lag
        if not np.any(mask):
            continue
        q_direct = z_centered[mask]
        keep_direct = np.abs(q_direct) > float(stage2_eps)
        if not np.any(keep_direct):
            continue
        y_tilde_direct = y_res[mask][keep_direct] / q_direct[keep_direct]
        w_direct = q_direct[keep_direct] ** 2
        direct_pred[:, lag] = _fit_weighted_stage2_predictor(
            X_obs[mask][keep_direct],
            y_tilde_direct,
            w_direct,
            panel.X,
            option=option,
            degree=stage2_degree,
            ridge_alpha=ridge_alpha,
            tree_max_depth=stage2_tree_max_depth,
            tree_min_leaf=stage2_tree_min_leaf,
            n_estimators=stage2_n_estimators,
            learning_rate=stage2_learning_rate,
            random_state=None if random_state is None else int(random_state) + 200000 + 10 * M + lag,
        )
    return {
        'warm_start': tau_metrics(warm_pred, target, sign_invariant=False),
        'direct_from_scratch': tau_metrics(direct_pred, target, sign_invariant=False),
    }
