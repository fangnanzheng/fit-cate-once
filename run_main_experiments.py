"""Main experiment runner, including lag-level Fisher aggregation over H0tl p-values."""
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy.special import gammaincc
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import cdm
import dm
import helpers
import rc
import ss
METHODS = ('Oracle', 'DM', 'cDM', 'SS')
DEFAULT_MAIN_COHORT_PROPS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
MAIN_N_SUBGROUPS = 5

def _counts_for_sample_size(N: int, T: int=5) -> List[int]:
    props = np.asarray(DEFAULT_MAIN_COHORT_PROPS, dtype=float)
    if T + 1 != props.size:
        raise ValueError(f'DEFAULT_MAIN_COHORT_PROPS has length {props.size}; expected T+1={T + 1}')
    raw = float(N) * props
    counts = np.floor(raw).astype(int)
    remainder = int(N) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)
        counts[order[:-remainder]] -= 1
    if np.any(counts < 0) or int(counts.sum()) != int(N):
        raise ValueError(f'Could not construct cohort counts for N={N}')
    return counts.astype(int).tolist()

def two_sided_pvalue(stat_obs: float, stats_perm: np.ndarray) -> float:
    stats_perm = np.abs(np.asarray(stats_perm, dtype=float))
    return float((1.0 + np.sum(stats_perm >= abs(float(stat_obs)))) / (len(stats_perm) + 1.0))

@dataclass
class ReplicateOutput:
    h0: List[dict]
    h0l: List[dict]
    h0lf: List[dict]
    h0tl: List[dict]
    h0k: List[dict]

def _combine_fisher_pvalues(pvalues: Sequence[float]) -> float:
    ps = np.clip(np.asarray(list(pvalues), dtype=float), 1e-300, 1.0)
    return float('nan') if ps.size == 0 else float(gammaincc(ps.size, float(-np.sum(np.log(ps)))))

def _row(
    rep: int,
    method: str,
    rmse: float,
    alpha: float,
    assumption: str,
    setting: str,
    n_perms: int,
    *,
    stat_obs: float | None=None,
    stats_perm: np.ndarray | None=None,
    pvalues: Sequence[float] | None=None,
    pv=None,
    **extra,
) -> dict:
    if pvalues is None:
        p = pv(stat_obs, stats_perm)
        out = {'stat_obs': stat_obs}
    else:
        seq = list(pvalues)
        p = _combine_fisher_pvalues(seq)
        out = {'n_terms': len(seq)}
    return {
        'rep': rep,
        'method': method,
        'rmse': rmse,
        **extra,
        **out,
        'pvalue': p,
        'reject': int(p <= alpha),
        'assumption': assumption,
        'setting': setting,
        'n_perms': n_perms,
        'alpha': alpha,
    }

def _run_one_replicate(
    rep: int,
    *,
    design: helpers.SADDesign,
    assumption: str,
    setting: str,
    option: str,
    n_perms: int,
    alpha: float,
    lags_for_h0tl: Sequence[int],
    base_seed: int,
    two_sided: bool,
    rc_mu_degree: int,
    rc_b_degree: int,
    rc_basis_degree: int,
    mu_ridge_alpha: float,
    rc_n_starts: int,
    rc_max_nfev: int,
    rc_convex_max_iters: int,
    ss_mu_degree: int,
    ss_basis_degree: int,
    ss_split_seed_offset: int,
    true_nuisances: bool=False,
) -> ReplicateOutput:
    rng = np.random.default_rng(base_seed + rep)
    T = design.T
    X = helpers.generate_covariates(design.N, rng)
    A = design.sample_A(rng)
    Y, mu0_true, tau_true = helpers.generate_outcomes(
        X,
        A,
        T,
        assumption,
        setting.lower() == 'validity',
        rng,
    )
    subgroups = helpers.make_subgroups_by_quantiles(X, K=MAIN_N_SUBGROUPS)
    true_mu = true_rv_b = true_rc_moment = true_ss_mhat = None
    if true_nuisances:
        true_mu = helpers.true_observed_mean_matrix(mu0_true, tau_true, design)
        true_rv_b = helpers.true_rv_second_moments(tau_true, design, assumption)
        true_rc_moment = helpers.true_rc_offdiag_moments(tau_true, design, assumption)
        true_ss_mhat = helpers.true_ss_mhat_dict(tau_true, design)
    method_rc = 'RV' if assumption == 'static' else 'RC'
    rc_res = (
        rc.fit_rv(
            X,
            Y,
            design,
            mu_degree=rc_mu_degree,
            b_degree=rc_b_degree,
            ridge_alpha=mu_ridge_alpha,
            option=option,
            assumption=assumption,
            true_mu=true_mu,
            true_b=true_rv_b,
        )
        if assumption == 'static'
        else rc.fit_rc(
            X,
            Y,
            design,
            mu_degree=rc_mu_degree,
            basis_degree=rc_basis_degree,
            ridge_alpha=mu_ridge_alpha,
            option=option,
            assumption=assumption,
            n_starts=rc_n_starts,
            max_nfev=rc_max_nfev,
            convex_max_iters=rc_convex_max_iters,
            random_state=base_seed + 10000 + rep,
            true_mu=true_mu,
            true_moment_target=true_rc_moment,
        )
    )
    cdm_res = cdm.fit_cdm(
        X,
        Y,
        mu_degree=rc_mu_degree,
        ridge_alpha=mu_ridge_alpha,
        option=option,
        assumption=assumption,
        true_mu=true_mu,
    )
    ss_res = ss.fit_ss_split(
        X,
        Y,
        A,
        design,
        mu_degree=ss_mu_degree,
        basis_degree=ss_basis_degree,
        mu_ridge_alpha=mu_ridge_alpha,
        split_seed=base_seed + ss_split_seed_offset + rep,
        option=option,
        assumption=assumption,
        true_mu=true_mu,
        true_mhat=true_ss_mhat,
    )

    def rmse_tau(tau_hat: np.ndarray, idx=None, lag=None) -> float:
        a, b = (tau_hat, tau_true) if idx is None else (tau_hat[idx], tau_true[idx])
        if lag is not None:
            a, b = (a[:, lag], b[:, lag])
        return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float('nan')
    test_idx = np.concatenate([split.test_idx for split in ss_res.splits])
    tau_ss_hat = ss_res.splits[0].comps.tau_hat
    rmse_global = {
        method_rc: rmse_tau(rc_res.tau_hat),
        'SS': rmse_tau(tau_ss_hat, idx=test_idx),
    }
    rmse_by_k = {
        method_rc: {
            k: rmse_tau(rc_res.tau_hat, idx=idx)
            for k, idx in subgroups.items()
        },
        'SS': {
            k: rmse_tau(
                tau_ss_hat,
                idx=np.intersect1d(idx, test_idx, assume_unique=False),
            )
            for k, idx in subgroups.items()
        },
    }

    def rmse_h0tl(method: str, t: int, l: int, k=None) -> float:
        if method not in {method_rc, 'SS'}:
            return float('nan')
        idx = np.where((A == t) | (A > t + l))[0]
        if k is not None:
            idx = np.intersect1d(idx, subgroups[k], assume_unique=False)
        if method == 'SS':
            idx = np.intersect1d(idx, test_idx, assume_unique=False)
        return rmse_tau(
            rc_res.tau_hat if method == method_rc else tau_ss_hat,
            idx=idx,
            lag=l,
        )

    def rmse_h0l(method: str, l: int) -> float:
        vals = [rmse_h0tl(method, t, l) for t in range(1, T - l + 1)]
        return float(np.nanmean(vals)) if method in {method_rc, 'SS'} else float('nan')
    rc_comps = helpers.build_likelihood_score_components(
        design,
        X,
        Y,
        rc_res.mu_models,
        rc_res.tau_hat,
    )
    oracle_comps = helpers.build_true_oracle_score_components(
        design,
        Y=Y,
        mu0_true=mu0_true,
        tau_true=tau_true,
    )
    pv = two_sided_pvalue if two_sided else helpers.right_tailed_pvalue
    A_perms_H0 = helpers.perms_H0(A, rng, n_perms)
    A_perms_H0_ss = ss.perms_H0_ss(ss_res, A, rng, n_perms)

    def global_results(subset=None):
        perms = A_perms_H0 if subset is None else helpers.perms_H_0k(A, subset, rng, n_perms)
        ss_perms = A_perms_H0_ss if subset is None else ss.perms_H_0k_ss(ss_res, A, subset, rng, n_perms)
        return [
            (
                method_rc,
                helpers.likelihood_score_stat_global_from_A(
                    A,
                    rc_comps,
                    subset=subset,
                ),
                helpers.likelihood_score_stats_global_from_perms(
                    perms,
                    rc_comps,
                    subset=subset,
                ),
            ),
            (
                'Oracle',
                helpers.likelihood_score_stat_global_from_A(
                    A,
                    oracle_comps,
                    subset=subset,
                ),
                helpers.likelihood_score_stats_global_from_perms(
                    perms,
                    oracle_comps,
                    subset=subset,
                ),
            ),
            (
                'DM',
                dm.observed_global(A, Y, design, subset=subset),
                dm.perms_global(perms, Y, design, subset=subset),
            ),
            (
                'cDM',
                cdm.observed_global(A, cdm_res.Y_adj, design, subset=subset),
                cdm.perms_global(perms, cdm_res.Y_adj, design, subset=subset),
            ),
            (
                'SS',
                (
                    ss.observed_global(ss_res, A)
                    if subset is None
                    else ss.observed_global_subgroup(ss_res, A, subset)
                ),
                (
                    ss.perms_global(ss_res, ss_perms)
                    if subset is None
                    else ss.perms_global_subgroup(ss_res, ss_perms, subset)
                ),
            ),
        ]
    h0 = [
        _row(
            rep,
            m,
            rmse_global.get(m, np.nan),
            alpha,
            assumption,
            setting,
            n_perms,
            stat_obs=s,
            stats_perm=sp,
            pv=pv,
        )
        for m, s, sp in global_results()
    ]
    h0k, h0l, h0lf, h0tl = ([], [], [], [])
    for k, idx in subgroups.items():
        for m, s, sp in global_results(idx):
            h0k.append(
                _row(
                    rep,
                    m,
                    rmse_by_k.get(m, {}).get(k, np.nan),
                    alpha,
                    assumption,
                    setting,
                    n_perms,
                    stat_obs=s,
                    stats_perm=sp,
                    pv=pv,
                    k=k,
                )
            )
    lag_methods = (method_rc, *METHODS)
    for l in [l for l in lags_for_h0tl if 0 <= l < T]:
        lag_obs = {m: 0.0 for m in lag_methods}
        lag_perm = {m: np.zeros(n_perms) for m in lag_methods}
        lag_pvalues: Dict[str, List[float]] = {m: [] for m in lag_methods}
        for t in range(1, T - l + 1):
            A_perms_tl = helpers.perms_H_tl(
                A,
                t=t,
                l=l,
                T=T,
                rng=rng,
                n_perms=n_perms,
            )
            A_perms_tl_ss = ss.perms_H_tl_ss(
                ss_res,
                A,
                t=t,
                l=l,
                T=T,
                rng=rng,
                n_perms=n_perms,
            )
            tl_results = [
                (
                    method_rc,
                    helpers.likelihood_score_stat_tl_from_A(
                        A,
                        rc_comps,
                        t=t,
                        l=l,
                    ),
                    helpers.likelihood_score_stats_tl_from_perms(
                        A_perms_tl,
                        rc_comps,
                        t=t,
                        l=l,
                    ),
                ),
                (
                    'Oracle',
                    helpers.likelihood_score_stat_tl_from_A(
                        A,
                        oracle_comps,
                        t=t,
                        l=l,
                    ),
                    helpers.likelihood_score_stats_tl_from_perms(
                        A_perms_tl,
                        oracle_comps,
                        t=t,
                        l=l,
                    ),
                ),
                (
                    'DM',
                    dm.observed_tl(A, Y, t=t, l=l),
                    dm.perms_tl(A_perms_tl, Y, t=t, l=l),
                ),
                (
                    'cDM',
                    cdm.observed_tl(A, cdm_res.Y_adj, t=t, l=l),
                    cdm.perms_tl(A_perms_tl, cdm_res.Y_adj, t=t, l=l),
                ),
                (
                    'SS',
                    ss.observed_tl(ss_res, A, t=t, l=l),
                    ss.perms_tl(ss_res, A_perms_tl_ss, t=t, l=l),
                ),
            ]
            for m, s, sp in tl_results:
                row = _row(
                    rep,
                    m,
                    rmse_h0tl(m, t, l),
                    alpha,
                    assumption,
                    setting,
                    n_perms,
                    stat_obs=s,
                    stats_perm=sp,
                    pv=pv,
                    t=t,
                    l=l,
                )
                h0tl.append(row)
                lag_obs[m] += float(s)
                lag_perm[m] += np.asarray(sp, dtype=float)
                lag_pvalues[m].append(float(row['pvalue']))
        denom = float(T - l)
        for m in lag_methods:
            h0l.append(
                _row(
                    rep,
                    m,
                    rmse_h0l(m, l),
                    alpha,
                    assumption,
                    setting,
                    n_perms,
                    stat_obs=lag_obs[m] / denom,
                    stats_perm=lag_perm[m] / denom,
                    pv=pv,
                    l=l,
                )
            )
            h0lf.append(
                _row(
                    rep,
                    m,
                    rmse_h0l(m, l),
                    alpha,
                    assumption,
                    setting,
                    n_perms,
                    pvalues=lag_pvalues[m],
                    l=l,
                )
            )
    return ReplicateOutput(h0=h0, h0l=h0l, h0lf=h0lf, h0tl=h0tl, h0k=h0k)
DETAIL_SPECS = {
    'h0': (
        [
            'rep',
            'method',
            'rmse',
            'stat_obs',
            'pvalue',
            'reject',
            'assumption',
            'setting',
            'n_perms',
            'alpha',
        ],
        ['method', 'assumption', 'setting'],
        ['method', 'assumption', 'setting', 'reject_rate', 'rmse', 'mean_p'],
        'H0',
    ),
    'h0l': (
        [
            'rep',
            'method',
            'rmse',
            'l',
            'stat_obs',
            'pvalue',
            'reject',
            'assumption',
            'setting',
            'n_perms',
            'alpha',
        ],
        ['method', 'l', 'assumption', 'setting'],
        [
            'method',
            'l',
            'assumption',
            'setting',
            'reject_rate',
            'rmse',
            'mean_p',
        ],
        'H0l',
    ),
    'h0lf': (
        [
            'rep',
            'method',
            'rmse',
            'l',
            'n_terms',
            'pvalue',
            'reject',
            'assumption',
            'setting',
            'n_perms',
            'alpha',
        ],
        ['method', 'l', 'assumption', 'setting'],
        [
            'method',
            'l',
            'assumption',
            'setting',
            'reject_rate',
            'rmse',
            'mean_p',
        ],
        'H0lF',
    ),
    'h0tl': (
        [
            'rep',
            'method',
            'rmse',
            't',
            'l',
            'stat_obs',
            'pvalue',
            'reject',
            'assumption',
            'setting',
            'n_perms',
            'alpha',
        ],
        ['method', 'l', 't', 'assumption', 'setting'],
        [
            'method',
            'l',
            't',
            'assumption',
            'setting',
            'reject_rate',
            'rmse',
            'mean_p',
        ],
        'H0tl',
    ),
    'h0k': (
        [
            'rep',
            'method',
            'rmse',
            'k',
            'stat_obs',
            'pvalue',
            'reject',
            'assumption',
            'setting',
            'n_perms',
            'alpha',
        ],
        ['method', 'k', 'assumption', 'setting'],
        [
            'method',
            'k',
            'assumption',
            'setting',
            'reject_rate',
            'rmse',
            'mean_p',
        ],
        'H0k',
    ),
}

def _save_outputs(outs: Sequence[ReplicateOutput], out_dir: str, meta: dict) -> None:
    for key, (detail_cols, group_cols, summary_cols, prefix) in DETAIL_SPECS.items():
        details = pd.DataFrame([row for out in outs for row in getattr(out, key)])[detail_cols]
        summary = (
            details.groupby(group_cols)
            .agg(
                reject_rate=('reject', 'mean'),
                rmse=('rmse', 'mean'),
                mean_p=('pvalue', 'mean'),
            )
            .reset_index()[summary_cols]
        )
        details.to_csv(os.path.join(out_dir, f'{prefix}_details.csv'), index=False)
        summary.to_csv(os.path.join(out_dir, f'{prefix}_summary.csv'), index=False)
    pd.Series(meta).to_json(os.path.join(out_dir, 'run_meta.json'), indent=2)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--assumption', type=str, choices=['static', 'lagged'], required=True)
    ap.add_argument('--setting', type=str, choices=['validity', 'power'], required=True)
    ap.add_argument('--option', type=str, choices=['parametric', 'nonparametric'], default='nonparametric')
    ap.add_argument('--n_reps', type=int, default=200)
    ap.add_argument('--n_perms', type=int, default=500)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--out_dir', type=str, default='results')
    ap.add_argument('--two_sided', action='store_true')
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--N', type=int, default=300)
    ap.add_argument('--rc_mu_degree', type=int, default=1)
    ap.add_argument('--rc_b_degree', type=int, default=1)
    ap.add_argument('--rc_basis_degree', type=int, default=1)
    ap.add_argument('--mu_ridge_alpha', type=float, default=0.001)
    ap.add_argument('--rc_n_starts', type=int, default=1)
    ap.add_argument('--rc_max_nfev', type=int, default=200)
    ap.add_argument('--rc_convex_max_iters', type=int, default=200)
    ap.add_argument('--ss_mu_degree', '--me_mu_degree', dest='ss_mu_degree', type=int, default=1)
    ap.add_argument('--ss_basis_degree', '--me_basis_degree', dest='ss_basis_degree', type=int, default=1)
    ap.add_argument('--h0tl_lags', type=str, default='0,1,2,3,4')
    ap.add_argument('--true_nuisances', action='store_true')
    args = ap.parse_args()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(THIS_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    counts = _counts_for_sample_size(args.N, T=args.T)
    lags_for_h0tl = [l for l in (int(x.strip()) for x in args.h0tl_lags.split(',') if x.strip()) if 0 <= l < args.T]
    design = helpers.SADDesign(T=args.T, counts_by_time=counts)
    from joblib import Parallel, delayed
    outs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        (
            delayed(_run_one_replicate)(
                rep,
                design=design,
                assumption=args.assumption,
                setting=args.setting,
                option=args.option,
                n_perms=args.n_perms,
                alpha=args.alpha,
                lags_for_h0tl=lags_for_h0tl,
                base_seed=args.seed,
                two_sided=args.two_sided,
                rc_mu_degree=args.rc_mu_degree,
                rc_b_degree=args.rc_b_degree,
                rc_basis_degree=args.rc_basis_degree,
                mu_ridge_alpha=args.mu_ridge_alpha,
                rc_n_starts=args.rc_n_starts,
                rc_max_nfev=args.rc_max_nfev,
                rc_convex_max_iters=args.rc_convex_max_iters,
                ss_mu_degree=args.ss_mu_degree,
                ss_basis_degree=args.ss_basis_degree,
                ss_split_seed_offset=50000,
                true_nuisances=bool(args.true_nuisances),
            )
            for rep in range(args.n_reps)
        )
    )
    _save_outputs(
        outs,
        out_dir,
        {
            'assumption': args.assumption,
            'setting': args.setting,
            'option': args.option,
            'n_reps': args.n_reps,
            'n_perms': args.n_perms,
            'alpha': args.alpha,
            'seed': args.seed,
            'two_sided': bool(args.two_sided),
            'T': args.T,
            'N': int(args.N),
            'cohort_props': list(DEFAULT_MAIN_COHORT_PROPS),
            'counts': counts,
            'n_subgroups': int(MAIN_N_SUBGROUPS),
            'h0tl_lags': lags_for_h0tl,
            'true_nuisances': bool(args.true_nuisances),
            'main_dgp': {
                'version': 'h2_sparse_positive_3p5_static_cov_unitvar',
                'responder_prob_scale': float(
                    helpers.MAIN_RESPONDER_PROB_SCALE
                ),
                'responder_probability': '0.08 * sigmoid(X1)',
                'positive_effect': float(helpers.MAIN_POSITIVE_EFFECT),
                'negative_effect': float(helpers.MAIN_NEGATIVE_EFFECT),
                'lagged_profile': list(helpers.MAIN_LAGGED_PROFILE),
                'subgroups': 'five empirical X1 quintiles',
            },
        },
    )
    print(f'Saved CSV outputs to: {out_dir}')
if __name__ == '__main__':
    main()
