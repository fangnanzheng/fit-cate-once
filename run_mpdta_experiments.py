from __future__ import annotations
import argparse
import json
from pathlib import Path
import helpers
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mpdta_exp_utils import (
    MPDTAData,
    REPORTED_LAGS,
    RIDGE_ALPHA,
    TREATED_FRAC_SUBGROUP,
    VALID_ASSUMPTIONS,
    VALID_NONPARAMETRIC_KINDS,
    VALID_STATISTICS,
    MPDTA_MU_GBRT_PARAMS,
    MPDTA_RV_MOMENT_GBRT_PARAMS,
    MPDTA_RC_MOMENT_GBRT_PARAMS,
    MPDTA_TREND_SCALE,
    STATIC_POWER_BASE_MAGNITUDE,
    STATIC_POWER_RANK_SLOPE,
    LAGGED_POWER_BASE_PROFILE,
    LAGGED_POWER_RANK_SLOPE_PROFILE,
    apply_tau,
    build_gate_tau_matrix,
    build_score_components,
    build_lag_tau_matrix,
    build_mu_matrix,
    build_subgroup_mu_matrix,
    default_effective_subgroup_tau,
    default_noise_scale,
    estimate_lag_nuisance,
    estimate_static_nuisances_by_source_t,
    fit_subgroup_focal_model,
    global_h0_pvalue,
    lag_h0l_pvalue,
    load_mpdta_data,
    normalize_nonparametric_kind,
    normalize_statistic,
    predict_subgroup_tau_hat,
    resolve_subgroup_tau_signs,
    sample_design_A,
    sample_sparse_lag_design_A,
    select_threshold_stump_from_tau_hat,
    subgroup_pvalue,
    subgroup_threshold_median,
    summarize_rejections,
    true_lag_observed_mean_matrix,
    true_rc_offdiag_moments_mpdta,
    true_rv_second_moments_mpdta,
    twfe_h0_pvalues,
    twfe_h0l_pvalues,
)
THIS_DIR = Path(__file__).resolve().parent
LAG_EXPERIMENTS = {'validity', 'power'}
METHOD = {'static': 'RV', 'lagged': 'RC'}
SUBGROUP_OOF_FOLDS = 5
SUBGROUP_MU_LEAF_GRID = (40, 60, 80, 100)
SUBGROUP_B_LEAF_GRID = (10, 20, 40)
SUBGROUP_THRESHOLD_METHOD = 'bagged_threshold_left_quantile'
SUBGROUP_STUMP_MIN_FRAC = 0.25
SUBGROUP_THRESHOLD_N_BAGS = 200
SUBGROUP_THRESHOLD_QUANTILE = 0.05

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Compact mpdta semi-synthetic experiments.')
    ap.add_argument(
        '--experiment',
        choices=[*LAG_EXPERIMENTS, 'subgroup'],
        required=True,
    )
    ap.add_argument('--assumption', choices=list(VALID_ASSUMPTIONS))
    ap.add_argument('--tau', type=float)
    ap.add_argument('--data_path', type=str, default=str(THIS_DIR / 'mpdta.csv'))
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--n_reps', type=int, default=100)
    ap.add_argument('--n_perms', type=int, default=300)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument(
        '--subgroup_statistic',
        choices=list(VALID_STATISTICS),
        default='aipw',
        help=(
            'Statistic for subgroup inference only; lag validity/power '
            'experiments remain likelihood by default.'
        ),
    )
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument(
        '--true_nuisance',
        action='store_true',
        help='For validity/power only, plug in oracle E[Y|X] and residual moments for RV/RC nuisance stages.',
    )
    ap.add_argument('--nonparametric_kind', choices=list(VALID_NONPARAMETRIC_KINDS), default='cart')
    ap.add_argument('--rv_mu_degree', type=int, default=1)
    ap.add_argument('--rv_b_degree', type=int, default=1)
    ap.add_argument('--rc_mu_degree', type=int, default=1)
    ap.add_argument('--rc_basis_degree', type=int, default=1)
    ap.add_argument('--rc_lambda_trace', type=float, default=0.0)
    ap.add_argument('--rc_nls_ridge', type=float, default=0.0)
    ap.add_argument('--subgroup_threshold_n_bags', type=int, default=SUBGROUP_THRESHOLD_N_BAGS)
    ap.add_argument('--subgroup_threshold_quantile', type=float, default=SUBGROUP_THRESHOLD_QUANTILE)
    return ap.parse_args()

def _lag_setting(experiment: str) -> str:
    value = str(experiment).strip().lower()
    if value not in LAG_EXPERIMENTS:
        raise ValueError(f'Expected a lag experiment, got {experiment!r}')
    return value

def _design_from_assignment(A: np.ndarray, T: int) -> helpers.SADDesign:
    counts = [int(np.sum(np.asarray(A, dtype=int) == t)) for t in range(1, T + 2)]
    return helpers.SADDesign(T=T, counts_by_time=counts)

def _result_row(
    rep: int,
    setting: str,
    assumption: str,
    method: str,
    pvalue: float,
    stat: float,
    alpha: float,
    *,
    hypothesis: str,
    lag: int | None=None,
    source_t: int | None=None,
) -> dict:
    return {
        'rep': rep,
        'setting': setting,
        'assumption': assumption,
        'hypothesis': hypothesis,
        'method': method,
        'lag': np.nan if lag is None else int(lag),
        'source_t': np.nan if source_t is None else int(source_t),
        'pvalue': float(pvalue),
        'reject': int(float(pvalue) <= float(alpha)),
        'stat': float(stat),
        'statistic': 'likelihood',
    }

def _eligible_summary_for_lag(A: np.ndarray, T: int, lag: int) -> dict:
    A = np.asarray(A, dtype=int)
    treated_counts = []
    control_counts = []
    eligible_counts = []
    for t in range(1, int(T) - int(lag) + 1):
        treated = int(np.sum(A == t))
        control = int(np.sum(A > t + int(lag)))
        treated_counts.append(treated)
        control_counts.append(control)
        eligible_counts.append(treated + control)
    return {
        'n_time_comparisons': int(len(eligible_counts)),
        'treated_count_min': int(np.min(treated_counts)) if treated_counts else 0,
        'control_count_min': int(np.min(control_counts)) if control_counts else 0,
        'eligible_count_min': int(np.min(eligible_counts)) if eligible_counts else 0,
        'treated_count_mean': float(np.mean(treated_counts)) if treated_counts else np.nan,
        'control_count_mean': float(np.mean(control_counts)) if control_counts else np.nan,
        'eligible_count_mean': float(np.mean(eligible_counts)) if eligible_counts else np.nan,
    }

def _tau_diagnostic_row(
    rep: int,
    setting: str,
    assumption: str,
    method: str,
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    *,
    lag: int | None=None,
    source_t: int | None=None,
    A: np.ndarray | None=None,
    T: int | None=None,
) -> dict:
    est = np.asarray(tau_hat, dtype=float).reshape(-1)
    tru = np.asarray(tau_true, dtype=float).reshape(-1)
    err = est - tru
    out = {
        'rep': int(rep),
        'setting': setting,
        'assumption': assumption,
        'method': method,
        'lag': np.nan if lag is None else int(lag),
        'source_t': np.nan if source_t is None else int(source_t),
        'tau_rmse': float(np.sqrt(np.mean(err ** 2))),
        'tau_bias': float(np.mean(err)),
        'mean_tau_hat': float(np.mean(est)),
        'mean_tau_true': float(np.mean(tru)),
        'mean_abs_tau_hat': float(np.mean(np.abs(est))),
        'mean_abs_tau_true': float(np.mean(np.abs(tru))),
        'tau_hat_sd': float(np.std(est)),
        'tau_true_sd': float(np.std(tru)),
    }
    mask = np.abs(tru) > 1e-12
    out['sign_alignment'] = float(np.mean(np.sign(est[mask]) == np.sign(tru[mask]))) if np.any(mask) else np.nan
    denom = float(np.linalg.norm(est) * np.linalg.norm(tru))
    out['cosine_alignment'] = float(np.dot(est, tru) / denom) if denom > 1e-12 else np.nan
    if A is not None and T is not None and lag is not None:
        out.update(_eligible_summary_for_lag(A, int(T), int(lag)))
    return out

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.nan
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _accuracy_metrics(est: np.ndarray, true: np.ndarray) -> dict:
    est = np.asarray(est, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    err = est - true
    denom = float(np.sqrt(np.mean(true ** 2)))
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'relative_rmse': float(np.sqrt(np.mean(err ** 2)) / denom) if denom > 1e-12 else np.nan,
        'bias': float(np.mean(err)),
        'mae': float(np.mean(np.abs(err))),
        'corr': _safe_corr(est, true),
        'mean_est': float(np.mean(est)),
        'mean_true': float(np.mean(true)),
        'sd_est': float(np.std(est)),
        'sd_true': float(np.std(true)),
    }

def _stage_row(
    rep: int,
    setting: str,
    assumption: str,
    method: str,
    *,
    stage: str,
    target: str,
    est: np.ndarray,
    true: np.ndarray,
    time: int | None=None,
    time_s: int | None=None,
    time_t: int | None=None,
) -> dict:
    out = {
        'rep': int(rep),
        'setting': setting,
        'assumption': assumption,
        'method': method,
        'stage': stage,
        'target': target,
        'time': np.nan if time is None else int(time),
        'time_s': np.nan if time_s is None else int(time_s),
        'time_t': np.nan if time_t is None else int(time_t),
    }
    out.update(_accuracy_metrics(est, true))
    return out

def _stage_diagnostic_rows(
    rep: int,
    setting: str,
    assumption: str,
    method: str,
    fit_result,
    true_mu: np.ndarray,
    true_moment: np.ndarray | None=None,
) -> list[dict]:
    rows: list[dict] = []
    if fit_result is None:
        return rows
    mu_hat = getattr(fit_result, 'mu_hat', None)
    if mu_hat is not None:
        mu_hat = np.asarray(mu_hat, dtype=float)
        true_mu = np.asarray(true_mu, dtype=float)
        rows.append(_stage_row(
                rep,
                setting,
                assumption,
                method,
                stage='mu',
                target='all_periods',
                est=mu_hat,
                true=true_mu,
            )
        )
        for t in range(true_mu.shape[1]):
            rows.append(_stage_row(
                    rep,
                    setting,
                    assumption,
                    method,
                    stage='mu',
                    target=f'period_{t + 1}',
                    est=mu_hat[:, t],
                    true=true_mu[:, t],
                    time=t + 1,
                )
            )
    if true_moment is not None:
        true_moment = np.asarray(true_moment, dtype=float)
        if assumption == 'static':
            est_moment = getattr(fit_result, 'b_hat', None)
            if est_moment is not None:
                est_moment = np.asarray(est_moment, dtype=float)
                rows.append(_stage_row(
                        rep,
                        setting,
                        assumption,
                        method,
                        stage='moment',
                        target='diagonal_all',
                        est=est_moment,
                        true=true_moment,
                    )
                )
                for t in range(true_moment.shape[1]):
                    rows.append(_stage_row(
                            rep,
                            setting,
                            assumption,
                            method,
                            stage='moment',
                            target=f'diagonal_t{t + 1}',
                            est=est_moment[:, t],
                            true=true_moment[:, t],
                            time=t + 1,
                        )
                    )
        else:
            est_moment = getattr(fit_result, 'moment_target', None)
            pairs = getattr(fit_result, 'moment_pairs', None)
            if est_moment is not None:
                est_moment = np.asarray(est_moment, dtype=float)
                rows.append(_stage_row(
                        rep,
                        setting,
                        assumption,
                        method,
                        stage='moment',
                        target='offdiag_all',
                        est=est_moment,
                        true=true_moment,
                    )
                )
                if pairs is None:
                    pairs = [(np.nan, np.nan)] * true_moment.shape[1]
                for j, pair in enumerate(pairs):
                    tt, ss = pair
                    rows.append(_stage_row(
                            rep,
                            setting,
                            assumption,
                            method,
                            stage='moment',
                            target=f'offdiag_t{int(tt)}_s{int(ss)}',
                            est=est_moment[:, j],
                            true=true_moment[:, j],
                            time_s=int(ss),
                            time_t=int(tt),
                        )
                    )
            raw_moment = getattr(fit_result, 'moment_raw', None)
            if raw_moment is not None:
                raw_moment = np.asarray(raw_moment, dtype=float)
                rows.append(_stage_row(
                        rep,
                        setting,
                        assumption,
                        method,
                        stage='moment_raw',
                        target='offdiag_raw_all',
                        est=raw_moment,
                        true=true_moment,
                    )
                )
    return rows

def _tau_arg(args: argparse.Namespace, data: MPDTAData) -> float:
    return float(args.tau) if args.tau is not None else default_effective_subgroup_tau(data)

def _one_lag_rep(rep: int, data: MPDTAData, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed + rep)
    setting = _lag_setting(args.experiment)
    assumption = args.assumption
    design, A = sample_sparse_lag_design_A(data, rng, assumption=assumption)
    noise_scale = default_noise_scale(setting, assumption=assumption)
    Y0 = build_mu_matrix(data, noise_scale=noise_scale, rng=rng, assumption=assumption)
    tau_true = build_lag_tau_matrix(data, assumption=assumption, setting=setting)
    true_mu = true_lag_observed_mean_matrix(data, design, tau_true, assumption=assumption)
    true_moment = (
        true_rv_second_moments_mpdta(
            data,
            tau_true,
            design,
            assumption=assumption,
            noise_scale=noise_scale,
        )
        if assumption == 'static'
        else true_rc_offdiag_moments_mpdta(
            data,
            tau_true,
            design,
            assumption=assumption,
            noise_scale=noise_scale,
        )
    )
    Y = apply_tau(Y0, A, tau_true)
    if assumption == 'static':
        nuisances = estimate_static_nuisances_by_source_t(
            design,
            data.X,
            Y,
            random_state=args.seed + (10000 if setting == 'validity' else 20000) + rep,
            rv_mu_degree=args.rv_mu_degree,
            rv_b_degree=args.rv_b_degree,
            ridge_alpha=RIDGE_ALPHA,
            true_mu=true_mu if args.true_nuisance else None,
            true_moment=true_moment if args.true_nuisance else None,
        )
        twfe_rows = twfe_h0_pvalues(Y, A)
        rows = []
        diagnostics = []
        stage_diagnostics = _stage_diagnostic_rows(
            rep,
            setting,
            assumption,
            METHOD[assumption],
            next(iter(nuisances.values())).fit_result if nuisances else None,
            true_mu,
            true_moment,
        )
        for source_t, nuisance in nuisances.items():
            pvalue, stat = global_h0_pvalue(nuisance.comps, A, rng=rng, n_perms=args.n_perms)
            rows.append(_result_row(
                    rep,
                    setting,
                    assumption,
                    METHOD[assumption],
                    pvalue,
                    stat,
                    args.alpha,
                    hypothesis='H0',
                    source_t=source_t,
                )
            )
            diagnostics.append(_tau_diagnostic_row(
                    rep,
                    setting,
                    assumption,
                    METHOD[assumption],
                    nuisance.tau_hat[:, 0],
                    tau_true[:, 0],
                    source_t=source_t,
                )
            )
            rows.extend(
                (
                    _result_row(
                        rep,
                        setting,
                        assumption,
                        tw.method,
                        tw.pvalue,
                        tw.stat,
                        args.alpha,
                        hypothesis='H0',
                        source_t=source_t,
                    )
                    for tw in twfe_rows
                )
            )
        return {'tests': rows, 'diagnostics': diagnostics, 'stage_diagnostics': stage_diagnostics}
    nuis = estimate_lag_nuisance(
        design,
        data.X,
        Y,
        assumption=assumption,
        random_state=args.seed + (10000 if setting == 'validity' else 20000) + rep,
        rv_mu_degree=args.rv_mu_degree,
        rv_b_degree=args.rv_b_degree,
        rc_mu_degree=args.rc_mu_degree,
        rc_basis_degree=args.rc_basis_degree,
        ridge_alpha=RIDGE_ALPHA,
        rc_lambda_trace=args.rc_lambda_trace,
        rc_nls_ridge=args.rc_nls_ridge,
        true_mu=true_mu if args.true_nuisance else None,
        true_moment=true_moment if args.true_nuisance else None,
    )
    rows = []
    diagnostics = []
    stage_diagnostics = _stage_diagnostic_rows(
        rep,
        setting,
        assumption,
        METHOD[assumption],
        nuis.fit_result,
        true_mu,
        true_moment,
    )
    for lag in REPORTED_LAGS:
        pvalue, stat = lag_h0l_pvalue(nuis.comps, A, l=lag, rng=rng, n_perms=args.n_perms)
        rows.append(_result_row(
                rep,
                setting,
                assumption,
                METHOD[assumption],
                pvalue,
                stat,
                args.alpha,
                hypothesis='H0l',
                lag=lag,
            )
        )
        diagnostics.append(_tau_diagnostic_row(
                rep,
                setting,
                assumption,
                METHOD[assumption],
                nuis.tau_hat[:, lag],
                tau_true[:, lag],
                lag=lag,
                A=A,
                T=design.T,
            )
        )
        rows.extend(
            (
                _result_row(
                    rep,
                    setting,
                    assumption,
                    tw.method,
                    tw.pvalue,
                    tw.stat,
                    args.alpha,
                    hypothesis='H0l',
                    lag=lag,
                )
                for tw in twfe_h0l_pvalues(Y, A, l=lag)
            )
        )
    return {'tests': rows, 'diagnostics': diagnostics, 'stage_diagnostics': stage_diagnostics}

def _make_nonparametric_params(kind: str, mu_leaf: int, b_leaf: int) -> tuple[dict, dict]:
    normalize_nonparametric_kind(kind)
    return ({'max_depth': 1, 'min_samples_leaf': int(mu_leaf)}, {'max_depth': 1, 'min_samples_leaf': int(b_leaf)})

def _tune_subgroup_leaves(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    A_train: np.ndarray,
    gate_train: np.ndarray,
    *,
    T: int,
    base_seed: int,
    nonparametric_kind: str,
) -> dict:
    n = X_train.shape[0]
    design_train = _design_from_assignment(A_train, T)
    fold_ids = np.empty(n, dtype=int)
    for pos, idx in enumerate(np.random.default_rng(base_seed).permutation(n)):
        fold_ids[idx] = pos % SUBGROUP_OOF_FOLDS
    candidates = []
    for mu_leaf in SUBGROUP_MU_LEAF_GRID:
        for b_leaf in SUBGROUP_B_LEAF_GRID:
            if mu_leaf < b_leaf:
                continue
            mu_params, b_params = _make_nonparametric_params(nonparametric_kind, mu_leaf, b_leaf)
            tau_oof = np.full(n, np.nan, dtype=float)
            for fold in range(SUBGROUP_OOF_FOLDS):
                tr = np.where(fold_ids != fold)[0]
                va = np.where(fold_ids == fold)[0]
                if tr.size == 0 or va.size == 0:
                    continue
                design_fold = _design_from_assignment(A_train[tr], T)
                model = fit_subgroup_focal_model(
                    X_train[tr],
                    Y_train[tr],
                    random_state=base_seed + 1000 * mu_leaf + 10 * b_leaf + fold,
                    nonparametric_kind=nonparametric_kind,
                    mu_params=mu_params,
                    b_params=b_params,
                )
                tau_raw_va = np.asarray(predict_subgroup_tau_hat(model, X_train[va], design_train), dtype=float)
                tau_oof[va] = resolve_subgroup_tau_signs(
                    model,
                    X_train[va],
                    Y_train[va],
                    design_train,
                    tau_raw_va,
                )[:, 0]
            if np.isnan(tau_oof).any():
                continue
            stump_info = select_threshold_stump_from_tau_hat(gate_train, tau_oof, min_frac=SUBGROUP_STUMP_MIN_FRAC)
            candidates.append({
                'mu_leaf': int(mu_leaf),
                'b_leaf': int(b_leaf),
                'stump_info': stump_info,
                'stump_mse': float(stump_info['stump_mse']),
                'stump_gain': float(stump_info['stump_gain']),
                'stump_gap': float(stump_info['stump_gap']),
            })
    if not candidates:
        raise RuntimeError('No valid subgroup tuning candidates were produced.')
    chosen = min(
        candidates,
        key=lambda row: (row['stump_mse'], row['stump_gap'], -row['stump_gain'], row['mu_leaf'], row['b_leaf']),
    )
    return {'mu_leaf': int(chosen['mu_leaf']), 'b_leaf': int(chosen['b_leaf']), 'stump_info': chosen['stump_info']}

def _bagged_left_threshold(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    gate: np.ndarray,
    *,
    T: int,
    base_seed: int,
    nonparametric_kind: str,
    mu_params: dict,
    b_params: dict,
    n_bags: int,
    quantile: float,
) -> float:
    q = float(np.clip(float(quantile), 0.0, 1.0))
    n_bags = max(1, int(n_bags))
    n = int(X.shape[0])
    rng = np.random.default_rng(base_seed)
    bag_thresholds: list[float] = []
    for bag in range(n_bags):
        idx = rng.integers(0, n, size=n)
        design_bag = _design_from_assignment(A[idx], T)
        model = fit_subgroup_focal_model(
            X[idx],
            Y[idx],
            random_state=base_seed + bag,
            nonparametric_kind=nonparametric_kind,
            mu_params=mu_params,
            b_params=b_params,
        )
        tau_raw_bag = np.asarray(predict_subgroup_tau_hat(model, X[idx], design_bag), dtype=float)
        tau_hat_bag = resolve_subgroup_tau_signs(model, X[idx], Y[idx], design_bag, tau_raw_bag)[:, 0]
        stump_info = select_threshold_stump_from_tau_hat(gate[idx], tau_hat_bag, min_frac=SUBGROUP_STUMP_MIN_FRAC)
        bag_thresholds.append(float(stump_info['threshold_left']))
    return float(np.quantile(np.asarray(bag_thresholds, dtype=float), q))

def _subgroup_rep(rep: int, data: MPDTAData, args: argparse.Namespace):
    rng = np.random.default_rng(args.seed + rep)
    T = len(data.years)
    design_full, A = sample_design_A(data, rng, treated_frac=TREATED_FRAC_SUBGROUP)
    threshold = subgroup_threshold_median(data)
    gate = data.X_raw[data.gating_variable].to_numpy(dtype=float)
    tau_true = build_gate_tau_matrix(data, threshold=threshold, effective_tau=_tau_arg(args, data))
    Y0 = build_subgroup_mu_matrix(data, noise_scale=default_noise_scale('subgroup'), rng=rng)
    Y = apply_tau(Y0, A, tau_true)
    tune = _tune_subgroup_leaves(
        data.X,
        Y,
        A,
        gate,
        T=T,
        base_seed=args.seed + 20000 + rep,
        nonparametric_kind=args.nonparametric_kind,
    )
    mu_params, b_params = _make_nonparametric_params(args.nonparametric_kind, int(tune['mu_leaf']), int(tune['b_leaf']))
    model = fit_subgroup_focal_model(
        data.X,
        Y,
        random_state=args.seed + 30000 + rep,
        nonparametric_kind=args.nonparametric_kind,
        mu_params=mu_params,
        b_params=b_params,
    )
    tau_hat_raw_full = np.asarray(predict_subgroup_tau_hat(model, data.X, design_full), dtype=float)
    tau_hat_full = resolve_subgroup_tau_signs(model, data.X, Y, design_full, tau_hat_raw_full)
    tau_hat_1d = tau_hat_full[:, 0]
    est_thr = _bagged_left_threshold(
        data.X,
        Y,
        A,
        gate,
        T=T,
        base_seed=args.seed + 40000 + rep,
        nonparametric_kind=args.nonparametric_kind,
        mu_params=mu_params,
        b_params=b_params,
        n_bags=args.subgroup_threshold_n_bags,
        quantile=args.subgroup_threshold_quantile,
    )
    true_effective = gate > float(threshold)
    selected_effective = gate > float(est_thr)
    selected_null = ~selected_effective
    null_leakage = float(np.mean(true_effective[selected_null])) if np.any(selected_null) else np.nan
    effective_purity = float(np.mean(true_effective[selected_effective])) if np.any(selected_effective) else np.nan
    effective_coverage = float(np.sum(selected_effective & true_effective) / max(np.sum(true_effective), 1))
    threshold_error = float(est_thr - threshold)
    detail_df = pd.DataFrame({
        'rep': rep,
        'countyreal': data.unit_ids,
        'gate_value': gate,
        'true_tau': tau_true[:, 0],
        'tau_hat': tau_hat_1d,
        'true_effective_subgroup': true_effective.astype(int),
        'true_null_subgroup': (~true_effective).astype(int),
        'selected_effective_subgroup': selected_effective.astype(int),
        'selected_null_subgroup': selected_null.astype(int),
        'estimated_threshold': est_thr,
        'true_threshold': float(threshold),
        'threshold_error': threshold_error,
        'null_leakage': null_leakage,
        'effective_purity': effective_purity,
        'effective_coverage': effective_coverage,
        'selected_mu_min_samples_leaf': int(tune['mu_leaf']),
        'selected_b_min_samples_leaf': int(tune['b_leaf'])
    })
    subgroup_statistic = normalize_statistic(args.subgroup_statistic)
    comps = build_score_components(design_full, data.X, Y, model.mu_models, tau_hat_full, statistic=subgroup_statistic)
    rows = []
    subgroup_indices = {
        'effective_subgroup': np.where(selected_effective)[0],
        'null_subgroup': np.where(selected_null)[0],
    }
    for name, idx in subgroup_indices.items():
        test = subgroup_pvalue(
            comps,
            A,
            idx,
            statistic=subgroup_statistic,
            rng=np.random.default_rng(args.seed + 70000 + 100 * rep + (0 if name == 'effective_subgroup' else 1)),
            n_perms=args.n_perms,
        )
        rows.append({
            'rep': rep,
            'assumption': 'static',
            'subgroup': name,
            'method': test.method,
            'statistic': subgroup_statistic,
            'group_size': int(len(idx)),
            'true_threshold': float(threshold),
            'estimated_threshold': est_thr,
            'threshold_error': threshold_error,
            'null_leakage': null_leakage,
            'effective_purity': effective_purity,
            'effective_coverage': effective_coverage,
            'selected_mu_min_samples_leaf': int(tune['mu_leaf']),
            'selected_b_min_samples_leaf': int(tune['b_leaf']),
            'mean_true_tau': float(np.mean(tau_true[idx, 0])) if len(idx) else np.nan,
            'mean_tau_hat': float(np.mean(tau_hat_full[idx, 0])) if len(idx) else np.nan,
            'pvalue': float(test.pvalue),
            'reject': int(float(test.pvalue) <= float(args.alpha)),
            'stat': float(test.stat)
        })
    return (detail_df, rows)

def run_experiment(args: argparse.Namespace, data: MPDTAData):
    if args.experiment in LAG_EXPERIMENTS:
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            (
                delayed(_one_lag_rep)(rep, data, args)
                for rep in range(args.n_reps)
            )
        )
        return {
            'tests': pd.DataFrame([row for bundle in results for row in bundle['tests']]),
            'diagnostics': pd.DataFrame([row for bundle in results for row in bundle.get('diagnostics', [])]),
            'stage_diagnostics': pd.DataFrame(
                [
                    row
                    for bundle in results
                    for row in bundle.get('stage_diagnostics', [])
                ]
            )
        }
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        (
            delayed(_subgroup_rep)(rep, data, args)
            for rep in range(args.n_reps)
        )
    )
    return {
        'selection_detail': pd.concat([detail for detail, _ in results], ignore_index=True),
        'subgroup_tests': pd.DataFrame([row for _, rows in results for row in rows]),
    }

def write_outputs(df_or_bundle, args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = args.experiment
    stem = f'{experiment_name}_{args.assumption}' if experiment_name in LAG_EXPERIMENTS else experiment_name
    if experiment_name in LAG_EXPERIMENTS:
        test_df = df_or_bundle['tests']
        test_df.to_csv(out_dir / f'{stem}_raw.csv', index=False)
        summary_cols = [
            'setting',
            'assumption',
            'hypothesis',
            'source_t' if args.assumption == 'static' else 'lag',
            'statistic',
            'method',
        ]
        summarize_rejections(test_df, summary_cols).to_csv(out_dir / f'{stem}_summary.csv', index=False)
        diag_df = df_or_bundle.get('diagnostics', pd.DataFrame())
        if not diag_df.empty:
            diag_df.to_csv(out_dir / f'{stem}_diagnostics_raw.csv', index=False)
            diag_group_cols = ['setting', 'assumption', 'source_t' if args.assumption == 'static' else 'lag', 'method']
            numeric_cols = [
                c
                for c in diag_df.columns
                if (
                    c not in set(diag_group_cols)
                    | {'rep', 'setting', 'assumption', 'method'}
                    and pd.api.types.is_numeric_dtype(diag_df[c])
                )
            ]
            (
                diag_df.groupby(diag_group_cols, dropna=False)[numeric_cols]
                .mean()
                .reset_index()
                .to_csv(out_dir / f'{stem}_diagnostics_summary.csv', index=False)
            )
        stage_df = df_or_bundle.get('stage_diagnostics', pd.DataFrame())
        if not stage_df.empty:
            stage_df.to_csv(out_dir / f'{stem}_stage_diagnostics_raw.csv', index=False)
            stage_group_cols = ['setting', 'assumption', 'method', 'stage', 'target', 'time', 'time_s', 'time_t']
            numeric_cols = [
                c
                for c in stage_df.columns
                if (
                    c not in set(stage_group_cols) | {'rep'}
                    and pd.api.types.is_numeric_dtype(stage_df[c])
                )
            ]
            (
                stage_df.groupby(stage_group_cols, dropna=False)[numeric_cols]
                .mean()
                .reset_index()
                .to_csv(
                    out_dir / f'{stem}_stage_diagnostics_summary.csv',
                    index=False,
                )
            )
    else:
        detail_df = df_or_bundle['selection_detail']
        test_df = df_or_bundle['subgroup_tests']
        detail_df.to_csv(out_dir / f'{stem}_selection_raw.csv', index=False)
        test_df.to_csv(out_dir / f'{stem}_inference_raw.csv', index=False)
        summarize_rejections(
            test_df,
            ['subgroup', 'statistic', 'method'],
        ).to_csv(out_dir / f'{stem}_inference_summary.csv', index=False)
        detail_df.groupby('rep', as_index=False).agg(
            estimated_threshold=('estimated_threshold', 'first'),
            true_threshold=('true_threshold', 'first'),
            threshold_error=('threshold_error', 'first'),
            null_leakage=('null_leakage', 'first'),
            effective_purity=('effective_purity', 'first'),
            effective_coverage=('effective_coverage', 'first'),
            selected_mu_min_samples_leaf=('selected_mu_min_samples_leaf', 'first'),
            selected_b_min_samples_leaf=('selected_b_min_samples_leaf', 'first')
        ).to_csv(out_dir / f'{stem}_selection_summary.csv', index=False)
    config = {
        **vars(args),
        'experiment_normalized': experiment_name,
        'reported_lags': list(REPORTED_LAGS),
        'mpdta_mu_gbrt_params': dict(MPDTA_MU_GBRT_PARAMS),
        'mpdta_rv_moment_gbrt_params': dict(MPDTA_RV_MOMENT_GBRT_PARAMS),
        'mpdta_rc_moment_gbrt_params': dict(MPDTA_RC_MOMENT_GBRT_PARAMS),
        'mpdta_trend_scale': float(MPDTA_TREND_SCALE),
        'power_cate_score': 'W_i = 2 * ((rank(Z_i) - 0.5) / N) - 1',
        'static_power_base_magnitude': float(STATIC_POWER_BASE_MAGNITUDE),
        'static_power_rank_slope': float(STATIC_POWER_RANK_SLOPE),
        'lagged_power_base_profile': list(LAGGED_POWER_BASE_PROFILE),
        'lagged_power_rank_slope_profile': list(LAGGED_POWER_RANK_SLOPE_PROFILE),
        'subgroup_oof_folds': SUBGROUP_OOF_FOLDS,
        'subgroup_mu_leaf_grid': list(SUBGROUP_MU_LEAF_GRID),
        'subgroup_b_leaf_grid': list(SUBGROUP_B_LEAF_GRID),
        'subgroup_stump_min_frac': SUBGROUP_STUMP_MIN_FRAC,
        'subgroup_threshold_method': SUBGROUP_THRESHOLD_METHOD,
        'subgroup_threshold_n_bags': int(args.subgroup_threshold_n_bags),
        'subgroup_threshold_quantile': float(args.subgroup_threshold_quantile),
    }
    (out_dir / f'{stem}_config.json').write_text(json.dumps(config, indent=2, sort_keys=True))

def print_summary(df_or_bundle, args: argparse.Namespace) -> None:
    if args.experiment in LAG_EXPERIMENTS:
        test_df = df_or_bundle['tests']
        summary_cols = [
            'setting',
            'assumption',
            'hypothesis',
            'source_t' if args.assumption == 'static' else 'lag',
            'statistic',
            'method',
        ]
        print(summarize_rejections(test_df, summary_cols).to_string(index=False))
        return
    print(
        summarize_rejections(
            df_or_bundle['subgroup_tests'],
            ['subgroup', 'statistic', 'method'],
        ).to_string(index=False)
    )
    rep0 = df_or_bundle['selection_detail'][
        df_or_bundle['selection_detail']['rep']
        == int(df_or_bundle['selection_detail']['rep'].min())
    ].iloc[0]
    print(
        f"rep={int(rep0['rep'])}, true_threshold={float(rep0['true_threshold']):.4f}, "
        f"estimated_threshold={float(rep0['estimated_threshold']):.4f}, "
        f"null_leakage={float(rep0['null_leakage']):.4f}, "
        f"effective_purity={float(rep0['effective_purity']):.4f}"
    )

def main() -> None:
    args = parse_args()
    args.nonparametric_kind = normalize_nonparametric_kind(args.nonparametric_kind)
    args.subgroup_statistic = normalize_statistic(args.subgroup_statistic)
    data = load_mpdta_data(args.data_path)
    if args.experiment in LAG_EXPERIMENTS:
        if args.assumption is None:
            raise ValueError('--assumption is required for lag experiments.')
    elif args.assumption not in (None, 'static'):
        raise ValueError('subgroup only supports --assumption static.')
    else:
        args.assumption = 'static'
    if len(data.years) != 5:
        raise ValueError('This simplified code assumes the 5-period mpdta panel.')
    out = run_experiment(args, data)
    write_outputs(out, args)
    print_summary(out, args)
if __name__ == '__main__':
    main()
