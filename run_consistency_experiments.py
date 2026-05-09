from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List
import pandas as pd
from joblib import Parallel, delayed
import consistency_exp_utils as cu

def _raw_one_rep(
    rep: int,
    *,
    N: int,
    assumption: str,
    option: str,
    seed: int,
    T: int,
    cohort_props: List[float],
    noise_sd: float,
    mu_config_map: Dict[int, Dict[str, Any]],
    moment_config_map: Dict[int, Dict[str, Any]],
    mu_degree: int,
    moment_degree: int,
    basis_degree: int,
    ridge_alpha: float,
    n_starts: int,
    max_nfev: int,
    convex_max_iters: int,
    lambda_trace: float,
    lambda_trace_schedule: str,
    nls_ridge: float,
    nls_ridge_schedule: str,
    refine_nls: bool,
    use_convex_init: bool,
) -> Dict[str, Any]:
    panel = cu.generate_panel(
        N,
        assumption=assumption,
        option=option,
        seed=seed + rep,
        T=T,
        cohort_props=cohort_props,
        noise_sd=noise_sd,
        experiment='raw',
    )
    if option == 'parametric':
        mu_cfg: Dict[str, Any] = {}
        moment_cfg: Dict[str, Any] = {}
    else:
        mu_cfg = cu.config_for_N(mu_config_map, N)
        moment_cfg = cu.config_for_N(moment_config_map, N)
    mu_metrics = cu.evaluate_mu_once(
        panel,
        option=option,
        mu_degree=mu_degree,
        ridge_alpha=ridge_alpha,
        mu_gbrt_params=mu_cfg or None,
        random_state=seed + 10000 + rep,
    )
    moment_metrics = cu.evaluate_moment_once(
        panel,
        assumption=assumption,
        option=option,
        mu_degree=mu_degree,
        moment_degree=moment_degree,
        ridge_alpha=ridge_alpha,
        mu_gbrt_params=mu_cfg or None,
        moment_gbrt_params=moment_cfg or None,
        random_state=seed + 20000 + rep,
        noise_sd=noise_sd,
    )
    if lambda_trace_schedule == 'inv_sqrt_n':
        lambda_trace_eff = float(lambda_trace) / float(N) ** 0.5
    else:
        lambda_trace_eff = float(lambda_trace)
    if nls_ridge_schedule == 'inv_sqrt_n':
        nls_ridge_eff = float(nls_ridge) / float(N) ** 0.5
    else:
        nls_ridge_eff = float(nls_ridge)
    tau_metrics = cu.evaluate_tau_once(
        panel,
        assumption=assumption,
        option=option,
        mu_degree=mu_degree,
        moment_degree=moment_degree,
        basis_degree=basis_degree,
        ridge_alpha=ridge_alpha,
        mu_gbrt_params=mu_cfg or None,
        moment_gbrt_params=moment_cfg or None,
        random_state=seed + 30000 + rep,
        n_starts=n_starts,
        max_nfev=max_nfev,
        convex_max_iters=convex_max_iters,
        lambda_trace=lambda_trace_eff,
        nls_ridge=nls_ridge_eff,
        refine_nls=refine_nls,
        use_convex_init=use_convex_init,
    )
    row = {
        'rep': rep,
        'N': N,
        'assumption': assumption,
        'option': option,
        'experiment': 'raw',
        'dgp_name': panel.dgp_name,
        **mu_metrics.as_dict(prefix='mu_'),
        **moment_metrics.as_dict(prefix='moment_'),
        **tau_metrics.as_dict(prefix='tau_'),
    }
    if option == 'nonparametric':
        row['mu_config'] = str(mu_cfg)
        row['moment_config'] = str(moment_cfg)
    return row

def _save_raw_family(details: pd.DataFrame, *, prefix: str, out_dir: str) -> None:
    metric_cols = [c for c in details.columns if c.startswith(prefix + '_')]
    base_cols = ['N', 'assumption', 'option', 'experiment', 'dgp_name']
    rename = {
        f'{prefix}_rmse': 'rmse',
        f'{prefix}_nmse': 'nmse',
        f'{prefix}_mse': 'mse',
        f'{prefix}_variance': 'variance',
    }
    fam = details[base_cols + metric_cols].rename(columns=rename)
    fam.to_csv(os.path.join(out_dir, f'{prefix}_consistency_details.csv'), index=False)
    summary = fam.groupby(base_cols, as_index=False).agg(
        mean_rmse=('rmse', 'mean'),
        sd_rmse=('rmse', 'std'),
        mean_nmse=('nmse', 'mean'),
        sd_nmse=('nmse', 'std'),
        mean_mse=('mse', 'mean'),
        mean_variance=('variance', 'mean'),
    )
    summary.to_csv(os.path.join(out_dir, f'{prefix}_consistency_summary.csv'), index=False)

def _start_one_rep(
    rep: int,
    *,
    N: int,
    m_grid: List[int],
    assumption: str,
    option: str,
    seed: int,
    T: int,
    cohort_props: List[float],
    noise_sd: float,
    mu_config_map: Dict[int, Dict[str, Any]],
    moment_config_map: Dict[int, Dict[str, Any]],
    mu_degree: int,
    moment_degree: int,
    basis_degree: int,
    ridge_alpha: float,
    n_starts: int,
    max_nfev: int,
    convex_max_iters: int,
    lambda_trace: float,
    lambda_trace_schedule: str,
    nls_ridge: float,
    nls_ridge_schedule: str,
    refine_nls: bool,
    use_convex_init: bool,
    sign_threshold: float,
    stage2_degree: int,
    stage2_tree_max_depth: int,
    stage2_tree_min_leaf: int,
    stage2_n_estimators: int,
    stage2_learning_rate: float,
) -> List[Dict[str, Any]]:
    panel = cu.generate_panel(
        N,
        assumption=assumption,
        option=option,
        seed=seed + rep,
        T=T,
        cohort_props=cohort_props,
        noise_sd=noise_sd,
        experiment='start',
        sign_threshold=sign_threshold,
    )
    if option == 'parametric':
        mu_cfg: Dict[str, Any] = {}
        moment_cfg: Dict[str, Any] = {}
    else:
        mu_cfg = cu.config_for_N(mu_config_map, N)
        moment_cfg = cu.config_for_N(moment_config_map, N)
    if lambda_trace_schedule == 'inv_sqrt_n':
        lambda_trace_eff = float(lambda_trace) / float(N) ** 0.5
    else:
        lambda_trace_eff = float(lambda_trace)
    if nls_ridge_schedule == 'inv_sqrt_n':
        nls_ridge_eff = float(nls_ridge) / float(N) ** 0.5
    else:
        nls_ridge_eff = float(nls_ridge)
    import numpy as np
    subset_rng = np.random.default_rng(int(seed) + 100003 * int(rep) + 10007 * int(N))
    subset_order = subset_rng.permutation(N)
    tau_ws_full = cu.fit_tau_estimator(
        panel,
        assumption='lagged',
        option=option,
        mu_degree=mu_degree,
        basis_degree=basis_degree,
        ridge_alpha=ridge_alpha,
        mu_gbrt_params=mu_cfg or None,
        moment_gbrt_params=moment_cfg or None,
        random_state=seed + 30000 + rep,
        n_starts=n_starts,
        max_nfev=max_nfev,
        convex_max_iters=convex_max_iters,
        lambda_trace=lambda_trace_eff,
        nls_ridge=nls_ridge_eff,
        refine_nls=refine_nls,
        use_convex_init=use_convex_init,
    )
    f_hat_precomputed = np.abs(tau_ws_full)
    mu_models = cu.helpers.fit_mu_models(
        panel.X,
        panel.Y,
        degree=mu_degree,
        ridge_alpha=ridge_alpha,
        option=option,
        random_state=seed + 80000 + rep,
        gbrt_params=mu_cfg or None,
    )
    mu_hat_precomputed = cu.helpers.predict_mu(mu_models, panel.X)
    rows: List[Dict[str, Any]] = []
    for M in m_grid:
        metrics_map = cu.fit_warm_start_and_direct_for_overall_tau(
            panel,
            option=option,
            M=M,
            subset_order=subset_order,
            mu_degree=mu_degree,
            basis_degree=basis_degree,
            ridge_alpha=ridge_alpha,
            mu_gbrt_params=mu_cfg or None,
            moment_gbrt_params=moment_cfg or None,
            f_hat_precomputed=f_hat_precomputed,
            mu_hat_precomputed=mu_hat_precomputed,
            random_state=seed + 30000 + rep,
            n_starts=n_starts,
            max_nfev=max_nfev,
            convex_max_iters=convex_max_iters,
            lambda_trace=lambda_trace_eff,
            nls_ridge=nls_ridge_eff,
            refine_nls=refine_nls,
            use_convex_init=use_convex_init,
            stage2_degree=stage2_degree,
            stage2_tree_max_depth=stage2_tree_max_depth,
            stage2_tree_min_leaf=stage2_tree_min_leaf,
            stage2_n_estimators=stage2_n_estimators,
            stage2_learning_rate=stage2_learning_rate,
        )
        for method, metrics in metrics_map.items():
            rows.append({
                'rep': rep,
                'N': N,
                'M': int(M),
                'assumption': assumption,
                'option': option,
                'experiment': 'start',
                'method': method,
                'dgp_name': panel.dgp_name,
                **metrics.as_dict(prefix='tau_'),
            })
    return rows

def _save_start_outputs(details: pd.DataFrame, *, out_dir: str) -> None:
    details.to_csv(os.path.join(out_dir, 'tau_warm_start_details.csv'), index=False)
    summary = details.groupby(
        ['N', 'M', 'assumption', 'option', 'experiment', 'method', 'dgp_name'],
        as_index=False,
    ).agg(
        mean_rmse=('tau_rmse', 'mean'),
        sd_rmse=('tau_rmse', 'std'),
        mean_nmse=('tau_nmse', 'mean'),
        sd_nmse=('tau_nmse', 'std'),
        mean_mse=('tau_mse', 'mean'),
        mean_variance=('tau_variance', 'mean'),
    )
    summary.to_csv(os.path.join(out_dir, 'tau_warm_start_summary.csv'), index=False)

def main() -> None:
    ap = argparse.ArgumentParser(description='Run c5 consistency / warm-start experiments for RV/RC.')
    ap.add_argument('--assumption', choices=['static', 'lagged'], required=True)
    ap.add_argument('--option', choices=['parametric', 'nonparametric'], required=True)
    ap.add_argument('--experiment', choices=['raw', 'start'], default='raw')
    ap.add_argument('--n_grid', type=str, default=','.join(map(str, cu.DEFAULT_N_GRID)))
    ap.add_argument('--start_N', type=int, default=cu.DEFAULT_START_N)
    ap.add_argument('--m_grid', type=str, default=','.join(map(str, cu.DEFAULT_M_GRID)))
    ap.add_argument('--sign_threshold', type=float, default=cu.DEFAULT_START_SIGN_THRESHOLD)
    ap.add_argument('--stage2_degree', type=int, default=1)
    ap.add_argument('--stage2_tree_max_depth', type=int, default=cu.DEFAULT_START_STAGE2_MAX_DEPTH)
    ap.add_argument('--stage2_tree_min_leaf', type=int, default=cu.DEFAULT_START_STAGE2_MIN_LEAF)
    ap.add_argument('--stage2_n_estimators', type=int, default=cu.DEFAULT_START_STAGE2_N_ESTIMATORS)
    ap.add_argument('--stage2_learning_rate', type=float, default=cu.DEFAULT_START_STAGE2_LEARNING_RATE)
    ap.add_argument('--n_reps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--out_dir', type=str, default='consistency_results_c5')
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--cohort_props', type=str, default=','.join(map(str, cu.DEFAULT_COHORT_PROPS)))
    ap.add_argument('--noise_sd', type=float, default=cu.DEFAULT_NOISE_SD)
    ap.add_argument('--mu_degree', type=int, default=1)
    ap.add_argument('--moment_degree', type=int, default=1)
    ap.add_argument('--basis_degree', type=int, default=1)
    ap.add_argument('--ridge_alpha', type=float, default=0.0)
    ap.add_argument('--n_starts', type=int, default=5)
    ap.add_argument('--max_nfev', type=int, default=2000)
    ap.add_argument('--convex_max_iters', type=int, default=2000)
    ap.add_argument('--lambda_trace', type=float, default=1.0)
    ap.add_argument('--lambda_trace_schedule', choices=['fixed', 'inv_sqrt_n'], default='inv_sqrt_n')
    ap.add_argument('--nls_ridge', type=float, default=0.0)
    ap.add_argument('--nls_ridge_schedule', choices=['fixed', 'inv_sqrt_n'], default='fixed')
    ap.add_argument('--no_refine_nls', action='store_true')
    ap.add_argument('--no_convex_init', action='store_true')
    args = ap.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.experiment == 'start' and args.option != 'nonparametric':
        raise ValueError(
            '--experiment start currently supports only --option nonparametric, '
            'because the start DGP has x-dependent sign and both CATE '
            'warm-start and sign R-learner stages are nonparametric.'
        )
    cohort_props = cu.parse_float_list(args.cohort_props)
    if args.option == 'nonparametric':
        mu_config_map, moment_config_map = cu.built_in_nonparametric_config_maps(
            args.assumption,
            experiment=args.experiment,
        )
    else:
        mu_config_map, moment_config_map = ({}, {})
    if args.experiment == 'raw':
        Ns = cu.parse_int_list(args.n_grid)
        rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
            (
                delayed(_raw_one_rep)(
                    rep,
                    N=N,
                    assumption=args.assumption,
                    option=args.option,
                    seed=args.seed + 1000000 * i,
                    T=args.T,
                    cohort_props=cohort_props,
                    noise_sd=args.noise_sd,
                    mu_config_map=mu_config_map,
                    moment_config_map=moment_config_map,
                    mu_degree=args.mu_degree,
                    moment_degree=args.moment_degree,
                    basis_degree=args.basis_degree,
                    ridge_alpha=args.ridge_alpha,
                    n_starts=args.n_starts,
                    max_nfev=args.max_nfev,
                    convex_max_iters=args.convex_max_iters,
                    lambda_trace=args.lambda_trace,
                    lambda_trace_schedule=args.lambda_trace_schedule,
                    nls_ridge=args.nls_ridge,
                    nls_ridge_schedule=args.nls_ridge_schedule,
                    refine_nls=not args.no_refine_nls,
                    use_convex_init=not args.no_convex_init,
                )
                for i, N in enumerate(Ns)
                for rep in range(args.n_reps)
            )
        )
        details = pd.DataFrame(rows)
        details.to_csv(os.path.join(out_dir, 'all_stage_metrics_details.csv'), index=False)
        for prefix in ('mu', 'moment', 'tau'):
            _save_raw_family(details, prefix=prefix, out_dir=out_dir)
        meta = {
            'assumption': args.assumption,
            'option': args.option,
            'experiment': args.experiment,
            'dgp_name': cu.dgp_name(args.assumption, args.option, args.experiment),
            'lagged_dgp': cu.DEFAULT_LAGGED_DGP,
            'n_grid': Ns,
            'n_reps': args.n_reps,
            'seed': args.seed,
            'T': args.T,
            'cohort_props': cohort_props,
            'noise_sd': args.noise_sd,
            'mu_degree': args.mu_degree,
            'moment_degree': args.moment_degree,
            'basis_degree': args.basis_degree,
            'ridge_alpha': args.ridge_alpha,
            'n_starts': args.n_starts,
            'max_nfev': args.max_nfev,
            'convex_max_iters': args.convex_max_iters,
            'lambda_trace': args.lambda_trace,
            'lambda_trace_schedule': args.lambda_trace_schedule,
            'nls_ridge': args.nls_ridge,
            'nls_ridge_schedule': args.nls_ridge_schedule,
            'refine_nls': not args.no_refine_nls,
            'use_convex_init': not args.no_convex_init,
            'nonparametric_config_source': (
                'built_in_start_specific_defaults'
                if args.option == 'nonparametric' and args.experiment == 'start'
                else (
                    'built_in_fast_plateau_v3_defaults'
                    if args.option == 'nonparametric'
                    else 'null'
                )
            ),
            'mu_best_map': mu_config_map,
            'moment_best_map': moment_config_map,
        }
        cu.save_json(os.path.join(out_dir, 'run_meta.json'), meta)
        print(f'Saved raw consistency outputs to: {os.path.abspath(out_dir)}')
        return
    if args.assumption != 'lagged':
        raise ValueError("experiment='start' is implemented for assumption='lagged' only")
    N = int(args.start_N)
    m_grid = cu.parse_int_list(args.m_grid)
    rows_nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        (
            delayed(_start_one_rep)(
                rep,
                N=N,
                m_grid=m_grid,
                assumption=args.assumption,
                option=args.option,
                seed=args.seed,
                T=args.T,
                cohort_props=cohort_props,
                noise_sd=args.noise_sd,
                mu_config_map=mu_config_map,
                moment_config_map=moment_config_map,
                mu_degree=args.mu_degree,
                moment_degree=args.moment_degree,
                basis_degree=args.basis_degree,
                ridge_alpha=args.ridge_alpha,
                n_starts=args.n_starts,
                max_nfev=args.max_nfev,
                convex_max_iters=args.convex_max_iters,
                lambda_trace=args.lambda_trace,
                lambda_trace_schedule=args.lambda_trace_schedule,
                nls_ridge=args.nls_ridge,
                nls_ridge_schedule=args.nls_ridge_schedule,
                refine_nls=not args.no_refine_nls,
                use_convex_init=not args.no_convex_init,
                sign_threshold=args.sign_threshold,
                stage2_degree=args.stage2_degree,
                stage2_tree_max_depth=args.stage2_tree_max_depth,
                stage2_tree_min_leaf=args.stage2_tree_min_leaf,
                stage2_n_estimators=args.stage2_n_estimators,
                stage2_learning_rate=args.stage2_learning_rate,
            )
            for rep in range(args.n_reps)
        )
    )
    rows = [row for chunk in rows_nested for row in chunk]
    details = pd.DataFrame(rows)
    _save_start_outputs(details, out_dir=out_dir)
    meta = {
        'assumption': args.assumption,
        'option': args.option,
        'experiment': args.experiment,
        'dgp_name': cu.dgp_name(args.assumption, args.option, args.experiment),
        'lagged_dgp': cu.DEFAULT_LAGGED_DGP,
        'start_N': N,
        'm_grid': m_grid,
        'sign_threshold': args.sign_threshold,
        'stage2_degree': args.stage2_degree,
        'stage2_tree_max_depth': args.stage2_tree_max_depth,
        'stage2_tree_min_leaf': args.stage2_tree_min_leaf,
        'stage2_n_estimators': args.stage2_n_estimators,
        'stage2_learning_rate': args.stage2_learning_rate,
        'warm_start_description': (
            'The assignment-blind nonparametric RC first stage estimates a '
            'lag-0-normalized CATE representative on the full sample. The '
            'start DGP uses shared-easy lagged CATE shape multiplied by an '
            'x-dependent sign. The second stage uses revealed assignments only '
            'and a GBRT R-learner-style weighted regression to learn the '
            'missing sign function g(x), then applies sign(g_hat(x)) * '
            '|tau_hat_RC(x)| across all lags.'
        ),
        'direct_description': (
            'Direct R-learner trained from the same revealed assignments '
            'without a warm-start magnitude.'
        ),
        'n_reps': args.n_reps,
        'seed': args.seed,
        'T': args.T,
        'cohort_props': cohort_props,
        'noise_sd': args.noise_sd,
        'mu_degree': args.mu_degree,
        'moment_degree': args.moment_degree,
        'basis_degree': args.basis_degree,
        'ridge_alpha': args.ridge_alpha,
        'n_starts': args.n_starts,
        'max_nfev': args.max_nfev,
        'convex_max_iters': args.convex_max_iters,
        'lambda_trace': args.lambda_trace,
        'lambda_trace_schedule': args.lambda_trace_schedule,
        'nls_ridge': args.nls_ridge,
        'nls_ridge_schedule': args.nls_ridge_schedule,
        'refine_nls': not args.no_refine_nls,
        'use_convex_init': not args.no_convex_init,
        'nonparametric_config_source': (
            'built_in_start_specific_defaults'
            if args.option == 'nonparametric' and args.experiment == 'start'
            else (
                'built_in_fast_plateau_v3_defaults'
                if args.option == 'nonparametric'
                else 'null'
            )
        ),
        'mu_best_map': mu_config_map,
        'moment_best_map': moment_config_map,
    }
    cu.save_json(os.path.join(out_dir, 'run_meta.json'), meta)
    print(f'Saved warm-start outputs to: {os.path.abspath(out_dir)}')
if __name__ == '__main__':
    main()
