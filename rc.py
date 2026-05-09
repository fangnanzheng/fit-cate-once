"""rc.py

Residual-moment CATE estimators used by the CATE-assisted randomization tests.

Implements:
- RV/diagonal: assumption='static'  (lag-invariant CATE magnitude)
- RC/off-diagonal: assumption='lagged' (lagged CATE vector up to sign)

Both first-stage estimators only use (X, Y) and known design probabilities, and
never use realized assignment A. By default, their up-to-sign output is passed
through the proposal's assignment-free unit-level sign-resolution stage before
being returned as tau_hat.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from helpers import (
    SADDesign,
    fit_mu_models,
    make_regressor,
    predict_mu,
    resolve_unit_signs_from_residual_trajectory,
    wrap_fixed_mu_models,
)

@dataclass
class RCResult:
    mu_models: List
    tau_hat: np.ndarray
    tau_hat_pm: Optional[np.ndarray] = None
    sign_hat: Optional[np.ndarray] = None
    best_assignment: Optional[np.ndarray] = None
    sign_loss: Optional[np.ndarray] = None
    mu_hat: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    moment_raw: Optional[np.ndarray] = None
    moment_target: Optional[np.ndarray] = None
    moment_pairs: Optional[List[Tuple[int, int]]] = None
    b_hat: Optional[np.ndarray] = None

@dataclass
class RVTimeSpecificResult:
    mu_models: List
    tau_hat: np.ndarray
    tau_hat_by_t: np.ndarray
    tau_hat_pm: Optional[np.ndarray] = None
    tau_hat_by_t_pm: Optional[np.ndarray] = None
    sign_hat: Optional[np.ndarray] = None
    sign_hat_by_t: Optional[np.ndarray] = None
    best_assignment: Optional[np.ndarray] = None
    best_assignment_by_t: Optional[np.ndarray] = None
    sign_loss: Optional[np.ndarray] = None
    sign_loss_by_t: Optional[np.ndarray] = None
    mu_hat: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    b_hat: Optional[np.ndarray] = None
# Lighter residual-moment GBRTs for b(x) and off-diagonal moment fits.
DEFAULT_MOMENT_GBRT_PARAMS: Dict[str, object] = {
    'n_estimators': 300,
    'learning_rate': 0.03,
    'max_depth': 1,
    'subsample': 0.8,
    'min_samples_leaf': 10,
}
DEFAULT_RV_MOMENT_GBRT_PARAMS = dict(DEFAULT_MOMENT_GBRT_PARAMS)
DEFAULT_RC_MOMENT_GBRT_PARAMS = dict(DEFAULT_MOMENT_GBRT_PARAMS)

def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)

def _project_psd(M: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(_symmetrize(M))
    evals = np.maximum(evals, 0.0)
    return evecs * evals @ evecs.T

def _prox_psd_trace(M: np.ndarray, *, step: float, lam: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(_symmetrize(M))
    evals = np.maximum(evals - step * float(lam), 0.0)
    return _symmetrize(evecs * evals @ evecs.T)

def _pgd_psd_trace(
    init: np.ndarray,
    *,
    obj,
    grad,
    lam: float,
    max_iters: int,
    tol: float,
    eta_init: float,
    backtracking: bool,
    bt_max: int,
    bt_shrink: float,
    bt_growth: float,
) -> np.ndarray:
    M = _symmetrize(init)
    obj_prev = float(obj(M))
    eta = float(eta_init)
    for _ in range(int(max_iters)):
        G = _symmetrize(grad(M))
        eta_try = eta
        best_M, best_obj = (None, np.inf)
        for _ in range(max(1, int(bt_max) if backtracking else 1)):
            cand = _prox_psd_trace(M - eta_try * G, step=eta_try, lam=lam)
            cand_obj = float(obj(cand))
            if cand_obj < best_obj:
                best_M, best_obj = (cand, cand_obj)
            if not backtracking or cand_obj <= obj_prev:
                M_new, obj_new = (cand, cand_obj)
                eta_try *= float(bt_growth)
                break
            eta_try *= float(bt_shrink)
        else:
            M_new = M if best_M is None else best_M
            obj_new = obj_prev if not np.isfinite(best_obj) else best_obj
        rel = np.linalg.norm(M_new - M, ord='fro') / max(1.0, np.linalg.norm(M, ord='fro'))
        M, obj_prev, eta = (M_new, obj_new, eta_try)
        if rel < float(tol):
            break
    return _project_psd(M)

def _principal_factor(M: np.ndarray) -> Tuple[float, np.ndarray]:
    evals, evecs = np.linalg.eigh(_symmetrize(M))
    idx = int(np.argmax(evals))
    lam1 = float(evals[idx])
    u1 = evecs[:, idx]
    return (lam1, u1)


def _fit_mu_residuals(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    degree: int,
    ridge_alpha: float,
    option: str,
    assumption: Optional[str],
    random_state: Optional[int],
    gbrt_params: Optional[Dict[str, object]],
    nonparametric_kind: str='gbrt',
    true_mu: Optional[np.ndarray]=None,
) -> Tuple[List, np.ndarray]:
    if true_mu is not None:
        mu_hat = np.asarray(true_mu, dtype=float)
        return (wrap_fixed_mu_models(X, mu_hat), Y - mu_hat)
    mu_models = fit_mu_models(
        X,
        Y,
        degree=degree,
        ridge_alpha=ridge_alpha,
        option=option,
        assumption=assumption,
        nonparametric_kind=nonparametric_kind,
        random_state=random_state,
        gbrt_params=gbrt_params,
    )
    return (mu_models, Y - predict_mu(mu_models, X))

def _fit_columnwise_targets(
    X: np.ndarray,
    targets: np.ndarray,
    *,
    degree: int,
    ridge_alpha: float,
    option: str,
    random_state: Optional[int],
    gbrt_params: Optional[Dict[str, object]]=None,
) -> np.ndarray:
    out = np.zeros_like(targets, dtype=float)
    rs0 = None if random_state is None else int(random_state)
    for j in range(targets.shape[1]):
        reg = make_regressor(
            degree=degree,
            ridge_alpha=ridge_alpha,
            option=option,
            random_state=None if rs0 is None else rs0 + j,
            gbrt_params=gbrt_params,
        )
        reg.fit(X, targets[:, j])
        out[:, j] = reg.predict(X)
    return out

def _maybe_resolve_signs(
    tau_pm: np.ndarray,
    R: np.ndarray,
    design: SADDesign,
    *,
    resolve_sign: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    tau_pm = np.asarray(tau_pm, dtype=float)
    if not bool(resolve_sign):
        return (tau_pm, None, None, None)
    tau_signed, sign_hat, best_assignment, best_loss = resolve_unit_signs_from_residual_trajectory(design, R, tau_pm)
    return (tau_signed, sign_hat, best_assignment, best_loss)

def fit_rv_time_specific(
    X: np.ndarray,
    Y: np.ndarray,
    design: SADDesign,
    *,
    mu_degree: int=1,
    b_degree: int=1,
    ridge_alpha: float=0.001,
    option: str='parametric',
    assumption: str='static',
    mu_gbrt_params: Optional[Dict[str, object]]=None,
    b_gbrt_params: Optional[Dict[str, object]]=None,
    random_state: Optional[int]=0,
    denom_eps: float=1e-10,
    true_mu: Optional[np.ndarray]=None,
    true_b: Optional[np.ndarray]=None,
    resolve_sign: bool=True,
) -> RVTimeSpecificResult:
    """Fit RV under assumption='static' and return both averaged and t-specific static CATE estimates."""
    T = design.T
    if Y.shape[1] != T:
        raise ValueError('Y must have shape (N, T)')
    mu_models, R = _fit_mu_residuals(
        X,
        Y,
        degree=mu_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        assumption=assumption,
        random_state=random_state,
        gbrt_params=mu_gbrt_params,
        true_mu=true_mu,
    )
    b_hat = np.asarray(true_b, dtype=float) if true_b is not None else _fit_columnwise_targets(
        X,
        R ** 2,
        degree=b_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        random_state=random_state,
        gbrt_params=DEFAULT_RV_MOMENT_GBRT_PARAMS if b_gbrt_params is None else dict(b_gbrt_params),
    )
    v = np.array([design.pi_leq[t] * (1.0 - design.pi_leq[t]) for t in range(1, T + 1)], dtype=float)
    v_bar = float(v.mean())
    b_bar = b_hat.mean(axis=1, keepdims=True)
    tau2_by_t = np.zeros((X.shape[0], T), dtype=float)
    for t in range(T):
        denom = float(v[t] - v_bar)
        if abs(denom) >= float(denom_eps):
            tau2_by_t[:, t] = (b_hat[:, t] - b_bar[:, 0]) / denom
    tau_star_by_t = np.sqrt(np.maximum(tau2_by_t, 0.0))
    x = v - v_bar
    mask = np.abs(x) >= float(denom_eps)
    if not np.any(mask):
        tau_star = np.zeros(X.shape[0], dtype=float)
    else:
        x_m = x[mask]
        tau2 = (b_hat[:, mask] - b_bar) @ x_m / float(np.sum(x_m ** 2))
        tau_star = np.sqrt(np.maximum(tau2, 0.0))
    tau_pm = np.tile(tau_star[:, None], (1, T))
    tau_signed, sign_hat, best_assignment, sign_loss = _maybe_resolve_signs(
        tau_pm,
        R,
        design,
        resolve_sign=resolve_sign,
    )
    tau_by_t_signed = np.zeros_like(tau_star_by_t, dtype=float)
    sign_by_t = np.ones_like(tau_star_by_t, dtype=float)
    best_assignment_by_t = np.full_like(tau_star_by_t, fill_value=T + 1, dtype=int)
    sign_loss_by_t = np.full_like(tau_star_by_t, fill_value=np.nan, dtype=float)
    for j in range(T):
        tau_pm_j = np.tile(tau_star_by_t[:, [j]], (1, T))
        tau_signed_j, sign_j, best_a_j, loss_j = _maybe_resolve_signs(tau_pm_j, R, design, resolve_sign=resolve_sign)
        tau_by_t_signed[:, j] = tau_signed_j[:, 0]
        if sign_j is not None:
            sign_by_t[:, j] = sign_j
        if best_a_j is not None:
            best_assignment_by_t[:, j] = best_a_j
        if loss_j is not None:
            sign_loss_by_t[:, j] = loss_j
    return RVTimeSpecificResult(
        mu_models=mu_models,
        tau_hat=tau_signed,
        tau_hat_by_t=tau_by_t_signed,
        tau_hat_pm=tau_pm,
        tau_hat_by_t_pm=tau_star_by_t,
        sign_hat=sign_hat,
        sign_hat_by_t=sign_by_t if resolve_sign else None,
        best_assignment=best_assignment,
        best_assignment_by_t=best_assignment_by_t if resolve_sign else None,
        sign_loss=sign_loss,
        sign_loss_by_t=sign_loss_by_t if resolve_sign else None,
        mu_hat=np.asarray(Y - R, dtype=float),
        residuals=R,
        b_hat=b_hat,
    )

def fit_rv(
    X: np.ndarray,
    Y: np.ndarray,
    design: SADDesign,
    *,
    mu_degree: int=1,
    b_degree: int=1,
    ridge_alpha: float=0.001,
    option: str='parametric',
    assumption: str='static',
    mu_gbrt_params: Optional[Dict[str, object]]=None,
    b_gbrt_params: Optional[Dict[str, object]]=None,
    random_state: Optional[int]=0,
    denom_eps: float=1e-10,
    true_mu: Optional[np.ndarray]=None,
    true_b: Optional[np.ndarray]=None,
    resolve_sign: bool=True,
) -> RCResult:
    """Fit RV under assumption='static' using the averaged diagonal residual-moment estimator."""
    details = fit_rv_time_specific(
        X,
        Y,
        design,
        mu_degree=mu_degree,
        b_degree=b_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        assumption=assumption,
        mu_gbrt_params=mu_gbrt_params,
        b_gbrt_params=b_gbrt_params,
        random_state=random_state,
        denom_eps=denom_eps,
        true_mu=true_mu,
        true_b=true_b,
        resolve_sign=resolve_sign,
    )
    return RCResult(
        mu_models=details.mu_models,
        tau_hat=np.asarray(details.tau_hat, dtype=float),
        tau_hat_pm=details.tau_hat_pm,
        sign_hat=details.sign_hat,
        best_assignment=details.best_assignment,
        sign_loss=details.sign_loss,
        mu_hat=details.mu_hat,
        residuals=details.residuals,
        b_hat=details.b_hat,
    )

def fit_rc(
    X: np.ndarray,
    Y: np.ndarray,
    design: SADDesign,
    *,
    mu_degree: int=1,
    basis_degree: int=1,
    ridge_alpha: float=0.001,
    option: str='nonparametric',
    assumption: str='lagged',
    include_bias: bool=True,
    mu_gbrt_params: Optional[Dict[str, object]]=None,
    moment_denoise: bool=True,
    moment_gbrt_params: Optional[Dict[str, object]]=None,
    n_starts: int=1,
    max_nfev: int=200,
    random_state: Optional[int]=None,
    lambda_trace: float=0.01,
    convex_max_iters: int=200,
    convex_tol: float=1e-05,
    convex_eta_init: float=20.0,
    convex_backtracking: bool=True,
    convex_bt_max: int=20,
    convex_bt_shrink: float=0.5,
    convex_bt_growth: float=1.05,
    use_convex_init: bool=True,
    refine_nls: bool=True,
    nls_ridge: float=0.01,
    true_mu: Optional[np.ndarray]=None,
    true_moment_target: Optional[np.ndarray]=None,
    resolve_sign: bool=True,
) -> RCResult:
    """Fit RC under assumption='lagged' using off-diagonal residual moments."""
    import math
    from scipy.optimize import least_squares
    from sklearn.preprocessing import PolynomialFeatures
    T = design.T
    if Y.shape[1] != T:
        raise ValueError('Y must have shape (N, T)')
    rng = np.random.default_rng(random_state)
    mu_models, R = _fit_mu_residuals(
        X,
        Y,
        degree=mu_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        assumption=assumption,
        random_state=random_state,
        gbrt_params=mu_gbrt_params,
        true_mu=true_mu,
    )
    basis = PolynomialFeatures(degree=basis_degree, include_bias=bool(include_bias))
    Phi = basis.fit_transform(X)
    N, p = Phi.shape
    pT = p * T
    pi = design.pi
    pairs = [(t, s) for t in range(2, T + 1) for s in range(1, t)]
    num_pairs = len(pairs)
    Rprod_raw = np.column_stack([R[:, t - 1] * R[:, s - 1] for t, s in pairs])
    if true_moment_target is not None:
        moment_target = np.asarray(true_moment_target, dtype=float)
    elif not moment_denoise:
        moment_target = Rprod_raw
    elif option == 'nonparametric':
        moment_target = _fit_columnwise_targets(
            X,
            Rprod_raw,
            degree=1,
            ridge_alpha=ridge_alpha,
            option='nonparametric',
            random_state=None if random_state is None else int(random_state) + 10000,
            gbrt_params=DEFAULT_RC_MOMENT_GBRT_PARAMS if moment_gbrt_params is None else dict(moment_gbrt_params),
        )
    else:
        XtX = Phi.T @ Phi + float(ridge_alpha) * np.eye(p, dtype=float)
        try:
            moment_target = Phi @ np.linalg.solve(XtX, Phi.T @ Rprod_raw)
        except np.linalg.LinAlgError:
            moment_target = Rprod_raw

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
    Hs = np.stack([_symmetrize(_J(t - s) @ _D(s) - np.outer(_pi_trunc(t), _pi_trunc(s))) for t, s in pairs], axis=0)
    if option == 'nonparametric':
        J_pairs = float(num_pairs)

        def solve_B(c: np.ndarray) -> np.ndarray:

            def obj(B: np.ndarray) -> float:
                err = np.einsum('jab,ab->j', Hs, B) - c
                return float(np.mean(err ** 2) + float(lambda_trace) * np.trace(B))

            def grad(B: np.ndarray) -> np.ndarray:
                err = np.einsum('jab,ab->j', Hs, B) - c
                return 2.0 / J_pairs * np.tensordot(err, Hs, axes=(0, 0))
            return _pgd_psd_trace(
                np.zeros((T, T), dtype=float),
                obj=obj,
                grad=grad,
                lam=lambda_trace,
                max_iters=convex_max_iters,
                tol=convex_tol,
                eta_init=convex_eta_init,
                backtracking=convex_backtracking,
                bt_max=convex_bt_max,
                bt_shrink=convex_bt_shrink,
                bt_growth=convex_bt_growth,
            )
        tau_hat = np.zeros((N, T), dtype=float)
        if use_convex_init:
            for i in range(N):
                lam1, u1 = _principal_factor(solve_B(moment_target[i]))
                tau_hat[i] = np.zeros(T, dtype=float) if lam1 <= 0.0 or not np.isfinite(lam1) else math.sqrt(lam1) * u1
        if refine_nls:

            def res_tau(tau: np.ndarray, c: np.ndarray) -> np.ndarray:
                r = c - np.einsum('a,jab,b->j', tau, Hs, tau)
                return r if float(nls_ridge) <= 0.0 else np.concatenate([r, math.sqrt(float(nls_ridge)) * tau])

            def jac_tau(tau: np.ndarray) -> np.ndarray:
                Jmat = -2.0 * np.einsum('jab,b->ja', Hs, tau)
                return Jmat if float(nls_ridge) <= 0.0 else np.vstack([Jmat, math.sqrt(float(nls_ridge)) * np.eye(T)])
            for i in range(N):
                c_i = moment_target[i]
                inits = [tau_hat[i].copy()]
                while len(inits) < int(n_starts):
                    inits.append(tau_hat[i].copy() + rng.normal(loc=0.0, scale=1.0, size=T))
                best_x, best_cost = (None, np.inf)
                for x0 in inits[:int(n_starts)]:
                    try:
                        res = least_squares(
                            lambda z,
                            ci=c_i: res_tau(z, ci),
                            x0,
                            jac=jac_tau,
                            method='lm',
                            max_nfev=int(max_nfev),
                        )
                    except Exception:
                        continue
                    if np.isfinite(res.cost) and float(res.cost) < best_cost:
                        best_x, best_cost = (res.x, float(res.cost))
                tau_hat[i] = inits[0] if best_x is None else best_x
        tau_signed, sign_hat, best_assignment, sign_loss = _maybe_resolve_signs(
            tau_hat,
            R,
            design,
            resolve_sign=resolve_sign,
        )
        return RCResult(
            mu_models=mu_models,
            tau_hat=tau_signed,
            tau_hat_pm=tau_hat,
            sign_hat=sign_hat,
            best_assignment=best_assignment,
            sign_loss=sign_loss,
            mu_hat=np.asarray(Y - R, dtype=float),
            residuals=R,
            moment_raw=Rprod_raw,
            moment_target=moment_target,
            moment_pairs=list(pairs),
        )
    outer = np.einsum('ni,nj->nij', Phi, Phi)

    def pred_and_B(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M4 = M.reshape(T, p, T, p)
        B = np.einsum('ni,aibj,nj->nab', Phi, M4, Phi)
        return (np.einsum('nab,jab->nj', B, Hs), B)

    def obj_M(M: np.ndarray) -> float:
        pred, _ = pred_and_B(M)
        err = pred - moment_target
        return float(np.mean(err ** 2) + float(lambda_trace) * np.trace(M))

    def grad_M(M: np.ndarray) -> np.ndarray:
        pred, _ = pred_and_B(M)
        W = np.tensordot(pred - moment_target, Hs, axes=(1, 0))
        return (2.0 / (N * num_pairs) * np.einsum('nab,nij->aibj', W, outer)).reshape(pT, pT)
    beta_sp = None
    inits: List[np.ndarray] = []
    if use_convex_init:
        try:
            rng_convex = np.random.default_rng(random_state)
            beta_rand = rng_convex.normal(loc=0.0, scale=0.01, size=pT)
            M_hat = _pgd_psd_trace(
                np.outer(beta_rand, beta_rand),
                obj=obj_M,
                grad=grad_M,
                lam=lambda_trace,
                max_iters=convex_max_iters,
                tol=convex_tol,
                eta_init=convex_eta_init,
                backtracking=convex_backtracking,
                bt_max=convex_bt_max,
                bt_shrink=convex_bt_shrink,
                bt_growth=convex_bt_growth,
            )
            lam1, u1 = _principal_factor(M_hat)
            beta_sp = math.sqrt(lam1) * u1 if lam1 > 0.0 else np.zeros(pT, dtype=float)
            epsN = float(N) ** (-0.5)
            if float(np.linalg.norm(beta_sp)) < epsN:
                if not (lam1 > 0.0 and np.all(np.isfinite(u1))):
                    u1 = rng.normal(loc=0.0, scale=1.0, size=pT)
                    norm = float(np.linalg.norm(u1))
                    u1 = np.ones(pT, dtype=float) / math.sqrt(pT) if norm == 0.0 else u1 / norm
                base_init = epsN * u1
            else:
                base_init = beta_sp.copy()
            inits.append(base_init)
            while len(inits) < int(n_starts):
                inits.append(base_init + epsN * rng.normal(loc=0.0, scale=1.0, size=pT))
        except Exception:
            beta_sp = None
            inits = []
    while not inits and len(inits) < int(n_starts):
        inits.append(rng.normal(loc=0.0, scale=1.0, size=pT))

    def residuals(beta_flat: np.ndarray) -> np.ndarray:
        tau = Phi @ beta_flat.reshape(T, p).T
        r = (moment_target - np.einsum('na,jab,nb->nj', tau, Hs, tau)).reshape(-1, order='F')
        return r if float(nls_ridge) <= 0.0 else np.concatenate([r, math.sqrt(float(nls_ridge)) * beta_flat])

    def jacobian(beta_flat: np.ndarray) -> np.ndarray:
        tau = Phi @ beta_flat.reshape(T, p).T
        tmp = 2.0 * np.einsum('jab,nb->nja', Hs, tau)
        J = np.zeros((N * num_pairs, pT), dtype=float)
        for j in range(num_pairs):
            rows = slice(j * N, (j + 1) * N)
            for a in range(T):
                cols = slice(a * p, (a + 1) * p)
                J[rows, cols] = (-tmp[:, j, a])[:, None] * Phi
        return J if float(nls_ridge) <= 0.0 else np.vstack([J, math.sqrt(float(nls_ridge)) * np.eye(pT)])
    if not refine_nls:
        Beta_hat = beta_sp.reshape(T, p) if beta_sp is not None else np.zeros((T, p), dtype=float)
        tau_hat = Phi @ Beta_hat.T
    else:
        best_x, best_cost = (None, np.inf)
        for x0 in inits[:int(n_starts)]:
            try:
                res = least_squares(residuals, x0, jac=jacobian, method='lm', max_nfev=int(max_nfev))
            except Exception:
                continue
            if np.isfinite(res.cost) and float(res.cost) < best_cost:
                best_x, best_cost = (res.x.copy(), float(res.cost))
        if best_x is None:
            raise RuntimeError('RC NLS optimization failed for all initializations')
        tau_hat = Phi @ best_x.reshape(T, p).T
    tau_signed, sign_hat, best_assignment, sign_loss = _maybe_resolve_signs(
        tau_hat,
        R,
        design,
        resolve_sign=resolve_sign,
    )
    return RCResult(
        mu_models=mu_models,
        tau_hat=tau_signed,
        tau_hat_pm=tau_hat,
        sign_hat=sign_hat,
        best_assignment=best_assignment,
        sign_loss=sign_loss,
        mu_hat=np.asarray(Y - R, dtype=float),
        residuals=R,
        moment_raw=Rprod_raw,
        moment_target=moment_target,
        moment_pairs=list(pairs),
    )
fit_rcd = fit_rv
fit_rco = fit_rc
DEFAULT_RCD_MOMENT_GBRT_PARAMS = DEFAULT_MOMENT_GBRT_PARAMS
DEFAULT_RCO_MOMENT_GBRT_PARAMS = DEFAULT_MOMENT_GBRT_PARAMS
