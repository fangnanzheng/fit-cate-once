from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import re
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = Path(__file__).resolve().parent
OUT_DIR = DATA_ROOT / 'figures'
ALPHA = 0.1
RATE_Y_LIMITS = (-0.05, 1.05)
SIZE_Y_LIMITS = (-0.05, 0.55)
MAIN_ROOT = DATA_ROOT / 'results' / 'main'
CONSISTENCY_ROOT = DATA_ROOT / 'results' / 'consistency'

METHODS_STATIC = ['RV', 'cDM', 'DM', 'SS']
METHODS_LAG = ['RC', 'cDM', 'DM', 'SS']
METHODS_MPDTA_STATIC = ['RV', 'TWFE']
METHODS_MPDTA_LAG = ['RC', 'TWFE']
METHODS_MPDTA_SUBGROUP = ['RV']

METHOD_LABELS = {
    'RV': 'RT(CATE)', 'RC': 'RT(CATE)', 'SS': 'RT(SS)', 'DM': 'RT(DM)', 'cDM': 'RT(cDM)',
    'TWFE': 'TWFE',
}
METHOD_COLORS = {
    'RV': 'tab:blue', 'RC': 'tab:blue', 'cDM': 'tab:orange', 'DM': 'tab:green',
    'SS': 'gray', 'TWFE': 'tab:red',
}
METHOD_MARKERS = {
    'RV': 'o', 'RC': 'o', 'cDM': '^', 'DM': 's', 'SS': 'D', 'TWFE': 'P'
}
ERROR_BAR_MODE = 'stderr'
LINE_PLOT_LINEWIDTH = 2.2
LINE_PLOT_MARKERSIZE = 8

FIG3_SAMPLE_SIZE: int | None = 500
FIG45_SAMPLE_SIZE = 500
FISHER_LAG = 0
MPDTA_STATIC_SOURCE_T_ORDER = [1, 2, 3, 4, 5]
MPDTA_LAG_ORDER = [0, 1, 2, 3, 4]
MPDTA_TAU_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3]
MPDTA_PROFILE_TAUS = [0.2, 0.25, 0.3]
MPDTA_SUBGROUP_SELECTION_FILENAMES = ('subgroup_selection_raw.csv',)
MPDTA_SUBGROUP_SELECTION_SUMMARY_FILENAMES = ('subgroup_selection_summary.csv',)
MPDTA_SUBGROUP_INFERENCE_FILENAMES = ('subgroup_inference_raw.csv',)
MPDTA_SUBGROUP_SELECTION_REP = 0
CONSISTENCY_BRANCH_FILENAMES = {
    'nonparametric': 'fig18_consistency_nonparametric_tau_nmse_vs_N.pdf',
    'parametric': 'fig19_consistency_parametric_tau_nmse_vs_N.pdf',
}

WARM_START_METHOD_ORDER = ['warm_start', 'direct_from_scratch']
WARM_START_METHOD_LABELS = {
    'warm_start': 'Ours',
    'direct_from_scratch': 'R-learner',
}
WARM_START_METHOD_COLORS = {
    'warm_start': 'tab:blue',
    'direct_from_scratch': 'tab:purple',
}
FIG20_WARM_START_FILENAME = 'fig20_warm_start_tau_nmse_vs_M.pdf'


# -----------------------------
# Generic helpers
# -----------------------------
get_method_label = lambda m: METHOD_LABELS.get(m, m)
get_method_color = lambda m: METHOD_COLORS.get(m, METHOD_COLORS.get(get_method_label(m), 'tab:blue'))
get_method_marker = lambda m: METHOD_MARKERS.get(m, METHOD_MARKERS.get(get_method_label(m), 'o'))


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return pd.read_csv(path)


def read_json_required(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return json.loads(path.read_text())


def apply_filters(df: pd.DataFrame, filters: dict[str, object] | None = None) -> pd.DataFrame:
    out = df.copy()
    for col, value in (filters or {}).items():
        if col not in out.columns:
            raise KeyError(f'Column {col!r} not found. Available columns: {list(out.columns)}')
        out = out.loc[out[col].isin(list(value))] if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)) else out.loc[out[col] == value]
    return out.copy()


def summarize_binary_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    methods: list[str],
) -> pd.DataFrame:
    out = (
        df.loc[df['method'].isin(methods), group_cols + ['method', 'reject']]
        .groupby(group_cols + ['method'], as_index=False)
        .agg(mean=('reject', 'mean'), std=('reject', 'std'), n=('reject', 'size'))
    )
    out['stderr'] = out['std'] / np.sqrt(out['n'])
    return out


def check_methods(df: pd.DataFrame, methods: list[str], label: str) -> None:
    missing = [m for m in methods if m not in set(df['method'].unique())]
    if missing:
        raise ValueError(f'{label}: missing methods {missing}. Available methods: {sorted(df["method"].unique())}')


# -----------------------------
# Source helpers (dir or zip)
# -----------------------------

def _is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix == '.zip'


def list_source_files(source: Path) -> list[str]:
    if source.is_dir():
        return sorted(str(p.relative_to(source)).replace('\\', '/') for p in source.rglob('*') if p.is_file())
    if _is_zip(source):
        with zipfile.ZipFile(source) as zf:
            return sorted(n for n in zf.namelist() if not n.endswith('/'))
    return []


def source_has_file(source: Path, rel_path: str) -> bool:
    rel_path = rel_path.replace('\\', '/')
    if source.is_dir():
        return (source / rel_path).exists()
    if _is_zip(source):
        with zipfile.ZipFile(source) as zf:
            return rel_path in zf.namelist()
    return False


def first_existing_rel_path(source: Path, rel_paths: list[str] | tuple[str, ...]) -> str:
    for rel_path in rel_paths:
        if source_has_file(source, rel_path):
            return rel_path
    raise FileNotFoundError(f'None of these candidate files exists in {source}: {list(rel_paths)}')


def source_has_any_file(source: Path, rel_paths: list[str] | tuple[str, ...]) -> bool:
    return any(source_has_file(source, rel_path) for rel_path in rel_paths)


def read_source_csv_any(source: Path, rel_paths: list[str] | tuple[str, ...]) -> pd.DataFrame:
    return read_source_csv(source, first_existing_rel_path(source, rel_paths))


def read_source_csv(source: Path, rel_path: str) -> pd.DataFrame:
    rel_path = rel_path.replace('\\', '/')
    if source.is_dir():
        return read_csv_required(source / rel_path)
    if _is_zip(source):
        with zipfile.ZipFile(source) as zf:
            try:
                data = zf.read(rel_path)
            except KeyError as exc:
                raise FileNotFoundError(f'Missing file: {rel_path} inside {source}') from exc
        return pd.read_csv(BytesIO(data))
    raise FileNotFoundError(f'Unsupported source: {source}')


def read_source_json(source: Path, rel_path: str) -> dict:
    rel_path = rel_path.replace('\\', '/')
    if source.is_dir():
        return read_json_required(source / rel_path)
    if _is_zip(source):
        with zipfile.ZipFile(source) as zf:
            try:
                data = zf.read(rel_path)
            except KeyError as exc:
                raise FileNotFoundError(f'Missing file: {rel_path} inside {source}') from exc
        return json.loads(data.decode('utf-8'))
    raise FileNotFoundError(f'Unsupported source: {source}')


def resolve_mpdta_source(base: Path = DATA_ROOT) -> Path | None:
    explicit = [base / 'results' / 'mpdta', base / 'results' / 'mpdta.zip', base / 'mpdta_results.zip']
    if (base / 'results').exists():
        explicit.extend(sorted((base / 'results').glob('mpdta*.zip')))
    explicit.extend(sorted(base.glob('mpdta_results*.zip')))
    for path in explicit:
        if path.exists():
            return path
    return None


def discover_subgroup_dirs(source: Path, tau_values: list[float] | None = None) -> dict[float, str]:
    wanted = None if tau_values is None else {round(float(t), 10) for t in tau_values}
    mapping: dict[float, str] = {}
    pat = re.compile(r'(^|/)(subgroup\d+(?:\.\d+)?)/subgroup_config\.json$')
    for rel in list_source_files(source):
        m = pat.search(rel)
        if not m:
            continue
        folder = m.group(2)
        tau = None
        try:
            tau = float(read_source_json(source, rel).get('tau'))
        except Exception:
            try:
                tau = float(folder.replace('subgroup', ''))
            except ValueError:
                pass
        if tau is None:
            continue
        tau = round(float(tau), 10)
        if wanted is None or tau in wanted:
            mapping[tau] = folder
    return dict(sorted(mapping.items()))


# -----------------------------
# Main-results helpers (sample-size folders / zips)
# -----------------------------

def find_sample_size_sources(root: Path, prefix: str) -> list[tuple[int, Path]]:
    if not root.exists():
        return []
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)(?:\.zip)?$')
    chosen: dict[int, Path] = {}
    for path in root.iterdir():
        if not path.is_dir() and not _is_zip(path):
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        n = int(m.group(1))
        if n not in chosen or (chosen[n].suffix == '.zip' and path.is_dir()):
            chosen[n] = path
    return sorted(chosen.items())


def has_sample_size_sources(root: Path, prefix: str) -> bool:
    return bool(find_sample_size_sources(root, prefix))


def ensure_data_root(root: Path = MAIN_ROOT) -> Path:
    missing = [f'{p}{{N}}' for p in ['sta_valid', 'sta_power', 'lag_valid', 'lag_power'] if not has_sample_size_sources(root, p)]
    if missing:
        raise FileNotFoundError(f'Failed to find required data under {root}. Missing: {missing}')
    return root


def read_csv_from_source(source: Path, filename: str) -> pd.DataFrame:
    return read_source_csv(source, filename)


def resolve_sample_source(
    root: Path,
    prefix: str,
    sample_size: int | None = None,
) -> tuple[int, Path]:
    sources = find_sample_size_sources(root, prefix)
    if not sources:
        raise FileNotFoundError(f'No sources matching {prefix}{{N}} found under {root}')
    if sample_size is None:
        return sources[-1]
    for n, source in sources:
        if n == sample_size:
            return n, source
    raise FileNotFoundError(f'Failed to find {prefix}{sample_size} under {root}. Available sample sizes: {[n for n, _ in sources]}')


def summarize_metric_over_sample_sizes(
    root: Path,
    prefix: str,
    filename: str,
    methods: list[str],
    filters: dict[str, object] | None = None,
) -> pd.DataFrame:
    pieces = []
    for n, source in find_sample_size_sources(root, prefix):
        df = apply_filters(read_csv_from_source(source, filename), filters)
        if df.empty:
            raise ValueError(f'No rows remain in {filename} from {source} after filters={filters}')
        check_methods(df, methods, f'{source}:{filename}')
        s = summarize_binary_metric(df, [], methods)
        s['N'] = n
        pieces.append(s)
    out = pd.concat(pieces, ignore_index=True)
    out['method'] = pd.Categorical(out['method'], categories=methods, ordered=True)
    return out.sort_values(['N', 'method']).reset_index(drop=True)


# -----------------------------
# Plot helpers
# -----------------------------

def line_plot_by_x(
    summary: pd.DataFrame,
    x_col: str,
    methods: list[str],
    xlabel: str,
    ylabel: str,
    save_path: Path,
    *,
    is_size: bool = False,
    x_order: list[int] | list[float] | None = None,
    figsize: tuple[float, float] = (7, 7),
    label_fontsize: int = 24,
    tick_fontsize: int = 22,
    legend_fontsize: int = 18,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    legend_loc: str = 'upper right',
) -> None:
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    x_values = list(x_order) if x_order is not None else sorted(summary[x_col].dropna().unique().tolist())
    for method in methods:
        sub = summary.loc[summary['method'] == method].copy()
        if x_order is not None:
            sub['_x_order'] = pd.Categorical(sub[x_col], categories=x_values, ordered=True)
            sub = sub.sort_values('_x_order')
        else:
            sub = sub.sort_values(x_col)
        ax.errorbar(
            sub[x_col].to_numpy(), sub['mean'].to_numpy(), yerr=sub[ERROR_BAR_MODE].to_numpy(),
            label=get_method_label(method), color=get_method_color(method), marker=get_method_marker(method),
            linewidth=LINE_PLOT_LINEWIDTH, markersize=LINE_PLOT_MARKERSIZE, markerfacecolor='white', markeredgewidth=1.8,
            elinewidth=1.4, capsize=4, capthick=1.4,
        )
    if is_size:
        ax.axhline(ALPHA, color='black', linestyle='--', linewidth=2.0)
    ax.set_xticks(x_values)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=10)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ymax = float(summary['mean'].max()) if not summary.empty else 1.0
        ymax = min(1.0, max(0.18 if is_size else 0.65, 1.10 * max(ymax, ALPHA if is_size else ymax)))
        ax.set_ylim(0.0, ymax)
    if show_legend:
        ax.legend(loc=legend_loc, frameon=False, fontsize=legend_fontsize)
    ax.grid(False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def line_plot_over_sample_sizes(
    summary: pd.DataFrame,
    methods: list[str],
    ylabel: str,
    save_path: Path,
    **kwargs,
) -> None:
    line_plot_by_x(summary, 'N', methods, 'N', ylabel, save_path, x_order=sorted(summary['N'].dropna().unique().tolist()), **kwargs)


def grouped_bar_with_error(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[int],
    methods: list[str],
    ylabel: str,
    xlabel: str,
    save_path: Path,
    *,
    is_size: bool = False,
    legend_ncol: int = 4,
    y_limits: tuple[float, float] | None = None,
) -> None:
    check_methods(df, methods, str(save_path))
    summary = summarize_binary_metric(df, [group_col], methods)
    mean_wide = summary.pivot(index=group_col, columns='method', values='mean').reindex(group_order).loc[:, methods]
    err_wide = summary.pivot(index=group_col, columns='method', values=ERROR_BAR_MODE).reindex(group_order).loc[:, methods]
    x = np.arange(len(group_order))
    width = 0.82 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset, mean_wide[method].to_numpy(), width=width, label=get_method_label(method),
            color=get_method_color(method), edgecolor='black', linewidth=1.6,
            yerr=err_wide[method].to_numpy(), ecolor='gray', capsize=4, error_kw={'elinewidth': 1.4},
        )
    if is_size:
        ax.axhline(ALPHA, color='black', linestyle='--', linewidth=2.0)
    ax.set_ylim(*(y_limits if y_limits is not None else (
        0.0, min(1.05, max(0.18 if is_size else 0.65, 1.08 * float(np.nanmax((mean_wide + err_wide).to_numpy())))))))
    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=legend_ncol, frameon=True, fontsize=15)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mpdta_subgroup_selection_profile(
    source: Path,
    rel_path: str,
    save_path: Path,
    *,
    rep: int = MPDTA_SUBGROUP_SELECTION_REP,
    figsize: tuple[float, float] = (7, 7),
    label_fontsize: int = 18,
    tick_fontsize: int = 18,
    legend_fontsize: int = 18,
) -> None:
    df = read_source_csv(source, rel_path)
    rep_df = df.loc[df['rep'] == int(rep)].copy()
    if rep_df.empty:
        raise ValueError(f'rep={rep} is not available in {rel_path}. Available reps: {sorted(df["rep"].dropna().unique().tolist())[:10]}')
    rep_df = rep_df.sort_values('gate_value').reset_index(drop=True)
    x = rep_df['gate_value'].to_numpy(dtype=float)
    y_true = rep_df['true_tau'].to_numpy(dtype=float)
    threshold_est = float(rep_df['estimated_threshold'].iloc[0])
    train_df = rep_df.loc[rep_df['tau_hat'].notna()].copy()
    if train_df.empty:
        raise ValueError(f'{rel_path} does not contain any non-null tau_hat values for rep={rep}.')
    low_mask = train_df['gate_value'].to_numpy(dtype=float) <= threshold_est
    train_y = train_df['tau_hat'].to_numpy(dtype=float)
    low_mean = float(np.nanmean(train_y[low_mask])) if np.any(low_mask) else 0.0
    high_mean = float(np.nanmean(train_y[~low_mask])) if np.any(~low_mask) else 0.0
    y_hat_avg = np.where(x <= threshold_est, low_mean, high_mean)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y_true, linewidth=LINE_PLOT_LINEWIDTH, color='black', label='True CATE')
    ax.plot(x, y_hat_avg, linewidth=LINE_PLOT_LINEWIDTH, color='tab:blue', label=r'Avg. $\widehat{\mathrm{CATE}}$')
    ax.set_xlabel('Log population (lpop)', fontsize=label_fontsize)
    ax.set_ylabel('CATE', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    y_min = float(np.nanmin(np.r_[y_true, y_hat_avg, 0.0]))
    y_max = float(np.nanmax(np.r_[y_true, y_hat_avg, 0.0]))
    pad = max(0.02, 0.08 * max(y_max - y_min, 1e-6))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def consistency_style_line_plot(
    summary: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    y_label: str,
    save_path: Path,
    figsize: tuple[float, float] = (7, 7),
) -> None:
    df = summary.sort_values(x_col).copy()
    fig, ax = plt.subplots(figsize=figsize)

    # Cast the horizontal axis to numeric values before plotting so Fig. 18
    # does not show every available sample size after tick labels are reset.
    x = pd.to_numeric(df[x_col], errors='raise').to_numpy(dtype=float)
    ax.plot(
        x,
        df[y_col].to_numpy(),
        marker='o',
        linewidth=LINE_PLOT_LINEWIDTH,
        markersize=LINE_PLOT_MARKERSIZE,
    )

    # Keep the N-axis uncluttered. Figures 18--19 should show only the two
    # paper-scale sample-size ticks, matching the parametric panel.
    if x_col == 'N':
        sparse_ticks = [10000, 20000]
        ax.xaxis.set_major_locator(FixedLocator(sparse_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in sparse_ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(x_col, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# -----------------------------
# MPDTA summarizers
# -----------------------------

def summarize_mpdta_metric(
    source: Path,
    rel_path: str | list[str] | tuple[str, ...],
    methods: list[str],
    x_col: str,
) -> pd.DataFrame:
    if isinstance(rel_path, (list, tuple)):
        df = read_source_csv_any(source, rel_path)
        label = ' or '.join(rel_path)
    else:
        df = read_source_csv(source, rel_path)
        label = rel_path
    check_methods(df, methods, label)
    if 'reject' in df.columns:
        out = summarize_binary_metric(df, [x_col], methods)
    elif 'reject_rate' in df.columns:
        out = df.loc[df['method'].isin(methods)].copy().rename(columns={'reject_rate': 'mean'})
        n = out['n'].replace(0, np.nan).astype(float) if 'n' in out.columns else pd.Series(np.nan, index=out.index)
        out['std'] = np.sqrt(out['mean'].astype(float) * (1 - out['mean'].astype(float)))
        out['stderr'] = out['std'] / np.sqrt(n)
    else:
        raise KeyError(f'{label} has neither reject nor reject_rate columns. Available columns: {list(df.columns)}')
    out['method'] = pd.Categorical(out['method'], categories=methods, ordered=True)
    return out.sort_values([x_col, 'method']).reset_index(drop=True)


def summarize_mpdta_subgroup_taus(
    source: Path,
    subgroup_dirs: dict[float, str],
    subgroup: str,
    method: str = 'RV',
) -> pd.DataFrame:
    pieces = []
    for tau, folder in subgroup_dirs.items():
        df = apply_filters(
            read_source_csv_any(source, [f'{folder}/{name}' for name in MPDTA_SUBGROUP_INFERENCE_FILENAMES]),
            {'method': method, 'subgroup': subgroup},
        )
        if df.empty:
            raise ValueError(f'No rows remain for method={method!r}, subgroup={subgroup!r} in {folder}')
        s = summarize_binary_metric(df, [], [method])
        s['tau'] = tau
        pieces.append(s)
    out = pd.concat(pieces, ignore_index=True)
    out['method'] = pd.Categorical(out['method'], categories=[method], ordered=True)
    return out.sort_values(['tau', 'method']).reset_index(drop=True)


def summarize_mpdta_threshold_ci(source: Path, subgroup_dirs: dict[float, str]) -> pd.DataFrame:
    rows = []
    for tau, folder in subgroup_dirs.items():
        summary_rels = [f'{folder}/{name}' for name in MPDTA_SUBGROUP_SELECTION_SUMMARY_FILENAMES]
        raw_rels = [f'{folder}/{name}' for name in MPDTA_SUBGROUP_SELECTION_FILENAMES]
        if source_has_any_file(source, summary_rels):
            df = read_source_csv_any(source, summary_rels)
        elif source_has_any_file(source, raw_rels):
            raw = read_source_csv_any(source, raw_rels)
            df = (
                raw.groupby('rep', as_index=False)
                .agg(estimated_threshold=('estimated_threshold', 'first'), true_threshold=('true_threshold', 'first'))
            )
        else:
            raise FileNotFoundError(f'Missing subgroup selection summary/raw files in {folder}')
        df = df.dropna(subset=['estimated_threshold'])
        if df.empty:
            raise ValueError(f'No non-null estimated_threshold values found for tau={tau} in {folder}')
        est = df['estimated_threshold'].to_numpy(dtype=float)
        true_vals = df['true_threshold'].dropna().to_numpy(dtype=float) if 'true_threshold' in df.columns else np.array([])
        rows.append({
            'tau': tau,
            'mean': float(np.mean(est)),
            'std': float(np.std(est, ddof=1)) if est.size > 1 else 0.0,
            'n': int(est.size),
            'ci_low': float(np.quantile(est, 0.025)),
            'ci_high': float(np.quantile(est, 0.975)),
            'true_threshold': float(true_vals[0]) if true_vals.size else np.nan,
        })
    out = pd.DataFrame(rows).sort_values('tau').reset_index(drop=True)
    out['stderr'] = out['std'] / np.sqrt(out['n'])
    return out


def plot_mpdta_threshold_ci(
    summary: pd.DataFrame,
    save_path: Path,
    *,
    figsize: tuple[float, float] = (7, 7),
    label_fontsize: int = 28,
    tick_fontsize: int = 28,
) -> None:
    df = summary.sort_values('tau').reset_index(drop=True)
    x = df['tau'].to_numpy(dtype=float)
    y = df['mean'].to_numpy(dtype=float)
    yerr = np.vstack([y - df['ci_low'].to_numpy(dtype=float), df['ci_high'].to_numpy(dtype=float) - y])
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.errorbar(
        x, y, yerr=yerr, color=get_method_color('RV'), marker=get_method_marker('RV'),
        linewidth=LINE_PLOT_LINEWIDTH, markersize=LINE_PLOT_MARKERSIZE, markerfacecolor='white', markeredgewidth=1.8,
        elinewidth=1.4, capsize=4, capthick=1.4,
    )
    true_vals = df['true_threshold'].dropna().to_numpy(dtype=float)
    if true_vals.size:
        ax.axhline(float(true_vals[0]), color='black', linestyle='--', linewidth=2.0)
    ax.set_xticks(x.tolist())
    ax.set_xlabel(r'$\tau_*$', fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel('Estimated threshold', fontsize=label_fontsize, labelpad=10)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    y_min = float(np.nanmin(np.r_[df['ci_low'].to_numpy(dtype=float), true_vals])) if true_vals.size else float(
        np.nanmin(df['ci_low'].to_numpy(dtype=float)))
    y_max = float(np.nanmax(np.r_[df['ci_high'].to_numpy(dtype=float), true_vals])) if true_vals.size else float(
        np.nanmax(df['ci_high'].to_numpy(dtype=float)))
    pad = max(1e-3, 0.08 * max(y_max - y_min, 1e-6))
    ax.set_ylim(y_min - pad, y_max + pad)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Consistency helpers
# -----------------------------

def _candidate_consistency_sources(base_root: Path, branch: str) -> list[Path]:
    branch = str(branch).strip().lower()
    candidates = [CONSISTENCY_ROOT / branch, CONSISTENCY_ROOT / f'{branch}.zip', base_root / branch, base_root / f'{branch}.zip']
    if branch == 'nonparametric':
        candidates.extend([CONSISTENCY_ROOT / 'nonparameric.zip', base_root / 'nonparameric.zip'])
    return list(dict.fromkeys(candidates))


def resolve_consistency_source(
    base_root: Path,
    branch: str,
    required_filename: str = 'tau_consistency_details.csv',
) -> Path:
    for path in _candidate_consistency_sources(base_root, branch):
        if path.is_dir() and (path / required_filename).exists():
            return path
        if _is_zip(path):
            with zipfile.ZipFile(path) as zf:
                if required_filename in zf.namelist() or f'{branch}/{required_filename}' in zf.namelist():
                    return path
    raise FileNotFoundError(f'Failed to find {required_filename} for consistency branch={branch!r}. Tried: {[str(p) for p in _candidate_consistency_sources(base_root, branch)]}')


def has_consistency_source(
    base_root: Path,
    branch: str,
    required_filename: str = 'tau_consistency_details.csv',
) -> bool:
    try:
        resolve_consistency_source(base_root, branch, required_filename)
        return True
    except Exception:
        return False


def read_consistency_details(base_root: Path, branch: str, prefix: str = 'tau') -> pd.DataFrame:
    filename = f'{prefix}_consistency_details.csv'
    source = resolve_consistency_source(base_root, branch, required_filename=filename)
    rel = filename if source.is_dir() else (filename if source_has_file(source, filename) else f'{branch}/{filename}')
    df = read_source_csv(source, rel)
    if df.empty:
        raise ValueError(f'{source}:{filename} is empty')
    return df.copy()


def summarize_consistency_details(details: pd.DataFrame) -> pd.DataFrame:
    required = {'N', 'assumption', 'option', 'dgp_name', 'rmse', 'nmse', 'mse', 'variance'}
    missing = required.difference(details.columns)
    if missing:
        raise KeyError(f'Consistency details are missing required columns {sorted(missing)}. Available columns: {list(details.columns)}')
    return (
        details.groupby(['N', 'assumption', 'option', 'dgp_name'], as_index=False)
        .agg(mean_rmse=('rmse', 'mean'), sd_rmse=('rmse', 'std'), mean_nmse=('nmse', 'mean'),
             sd_nmse=('nmse', 'std'), mean_mse=('mse', 'mean'), mean_variance=('variance', 'mean'))
        .sort_values(['N', 'assumption', 'option', 'dgp_name']).reset_index(drop=True)
    )


def _candidate_warm_start_sources(base_root: Path) -> list[Path]:
    """Candidate locations for the warm-start consistency experiment."""
    return list(dict.fromkeys([
        CONSISTENCY_ROOT / 'start',
        CONSISTENCY_ROOT / 'start.zip',
        base_root / 'results' / 'consistency' / 'start',
        base_root / 'results' / 'consistency' / 'start.zip',
        base_root / 'start',
        base_root / 'start.zip',
    ]))


def resolve_warm_start_source(
    base_root: Path,
    required_filename: str = 'tau_warm_start_summary.csv',
) -> Path:
    for path in _candidate_warm_start_sources(base_root):
        if path.is_dir() and (path / required_filename).exists():
            return path
        if _is_zip(path):
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
                if required_filename in names or f'start/{required_filename}' in names:
                    return path
    raise FileNotFoundError(
        f'Failed to find {required_filename} for warm-start consistency results. '
        f'Tried: {[str(p) for p in _candidate_warm_start_sources(base_root)]}'
    )


def has_warm_start_source(
    base_root: Path,
    required_filename: str = 'tau_warm_start_summary.csv',
) -> bool:
    try:
        resolve_warm_start_source(base_root, required_filename=required_filename)
        return True
    except Exception:
        return False


def read_warm_start_csv(base_root: Path, filename: str) -> pd.DataFrame:
    source = resolve_warm_start_source(base_root, required_filename=filename)
    rel = filename if source.is_dir() else (filename if source_has_file(source, filename) else f'start/{filename}')
    df = read_source_csv(source, rel)
    if df.empty:
        raise ValueError(f'{source}:{filename} is empty')
    return df.copy()


def summarize_warm_start_details(details: pd.DataFrame) -> pd.DataFrame:
    # c5 writes tau_* columns in the details file. We also accept unprefixed names for compatibility.
    if 'tau_nmse' in details.columns:
        nmse_col = 'tau_nmse'
    elif 'nmse' in details.columns:
        nmse_col = 'nmse'
    else:
        raise KeyError(f'Warm-start details must contain tau_nmse or nmse. Available columns: {list(details.columns)}')

    if 'tau_rmse' in details.columns:
        rmse_col = 'tau_rmse'
    elif 'rmse' in details.columns:
        rmse_col = 'rmse'
    else:
        rmse_col = None

    required = {'M', 'method', nmse_col}
    missing = required.difference(details.columns)
    if missing:
        raise KeyError(f'Warm-start details are missing required columns {sorted(missing)}. Available columns: {list(details.columns)}')

    agg = {'mean_nmse': (nmse_col, 'mean'), 'sd_nmse': (nmse_col, 'std'), 'n': (nmse_col, 'size')}
    if rmse_col is not None:
        agg['mean_rmse'] = (rmse_col, 'mean')
        agg['sd_rmse'] = (rmse_col, 'std')
    out = details.groupby(['M', 'method'], as_index=False).agg(**agg)
    return out.sort_values(['M', 'method']).reset_index(drop=True)


def read_warm_start_summary(base_root: Path) -> pd.DataFrame:
    # Prefer the experiment summary. If it is missing, reconstruct the same quantity from details.
    try:
        summary = read_warm_start_csv(base_root, 'tau_warm_start_summary.csv')
    except FileNotFoundError:
        summary = summarize_warm_start_details(read_warm_start_csv(base_root, 'tau_warm_start_details.csv'))

    required = {'M', 'method', 'mean_nmse'}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f'Warm-start summary is missing required columns {sorted(missing)}. Available columns: {list(summary.columns)}')

    summary = summary.loc[summary['method'].isin(WARM_START_METHOD_ORDER)].copy()
    if summary.empty:
        raise ValueError(f'Warm-start summary contains none of the expected methods: {WARM_START_METHOD_ORDER}')
    summary['method'] = pd.Categorical(summary['method'], categories=WARM_START_METHOD_ORDER, ordered=True)
    return summary.sort_values(['M', 'method']).reset_index(drop=True)


def warm_start_consistency_style_line_plot(
    summary: pd.DataFrame,
    *,
    save_path: Path,
    figsize: tuple[float, float] = (7, 7),
) -> None:
    """Plot Fig. 20 in the same visual style as the consistency figures."""
    fig, ax = plt.subplots(figsize=figsize)

    x_values = sorted(summary['M'].dropna().unique().tolist())

    for method in WARM_START_METHOD_ORDER:
        sub = summary.loc[summary['method'] == method].sort_values('M')
        if sub.empty:
            continue

        ax.plot(
            sub['M'].to_numpy(),
            sub['mean_nmse'].to_numpy(),
            marker='o',
            color=WARM_START_METHOD_COLORS[method],
            label=WARM_START_METHOD_LABELS[method],
            linewidth=LINE_PLOT_LINEWIDTH,
            markersize=LINE_PLOT_MARKERSIZE,
        )

    tick_values = [x for x in [1600, 3200] if x in x_values]
    if not tick_values:
        tick_values = x_values[-2:]

    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(int(x)) for x in tick_values])

    ax.set_xlabel('M', fontsize=24)
    ax.set_ylabel('NMSE', fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    ax.legend(loc='upper right', frameon=False, fontsize=18)
    ax.grid(False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# -----------------------------
# Figure builders
# -----------------------------

def generate_original_figures(root: Path = MAIN_ROOT) -> list[Path]:
    root = ensure_data_root(root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    cfgs = [
        (1, 'sta_valid', 'H0_details.csv', METHODS_STATIC, None, 'Type I error rate',
         'fig1_sta_valid_sharp_null_size_continuous.pdf', True, SIZE_Y_LIMITS),
        (2, 'sta_power', 'H0_details.csv', METHODS_STATIC, None, 'Power',
         'fig2_sta_power_sharp_null_power_continuous.pdf', False, RATE_Y_LIMITS),
        (6, 'lag_valid', 'H0lF_details.csv', METHODS_LAG, {'l': FISHER_LAG}, 'Type I error rate',
         'fig6_lag_valid_fisher_l0_size_continuous.pdf', True, SIZE_Y_LIMITS),
        (7, 'lag_power', 'H0lF_details.csv', METHODS_LAG, {'l': FISHER_LAG}, 'Power',
         'fig7_lag_power_fisher_l0_power_continuous.pdf', False, RATE_Y_LIMITS),
    ]
    for _, prefix, filename, methods, filters, ylabel, out_name, is_size, y_limits in cfgs:
        summary = summarize_metric_over_sample_sizes(root, prefix, filename, methods, filters)
        path = OUT_DIR / out_name
        line_plot_over_sample_sizes(summary, methods, ylabel, path, is_size=is_size, figsize=(7, 7), label_fontsize=24,
                                    tick_fontsize=24, legend_fontsize=18, y_limits=y_limits, legend_loc='upper left')
        out.append(path)

    _, fig3_source = resolve_sample_source(root, 'sta_power', FIG3_SAMPLE_SIZE)
    fig3_df = read_csv_from_source(fig3_source, 'H0k_details.csv')
    fig3_path = OUT_DIR / 'fig3_sta_power_zero_subgroup_power.pdf'
    grouped_bar_with_error(fig3_df, 'k', [1, 2, 3, 4, 5], METHODS_STATIC, 'Power', 'Subgroup', fig3_path, y_limits=(0.0, 0.6))
    out.append(fig3_path)

    for prefix, filename, filters, ylabel, out_name, is_size, y_limits in [
        ('lag_valid', 'H0tl_details.csv', {'t': 1}, 'Type I error rate', 'fig4_lag_valid_t1_by_l_size.pdf', True, SIZE_Y_LIMITS),
        ('lag_power', 'H0tl_details.csv', {'t': 1}, 'Power', 'fig5_lag_power_t1_by_l_power.pdf', False, RATE_Y_LIMITS),
    ]:
        _, source = resolve_sample_source(root, prefix, FIG45_SAMPLE_SIZE)
        df = apply_filters(read_csv_from_source(source, filename), filters)
        summary = summarize_binary_metric(df, ['l'], METHODS_LAG)
        path = OUT_DIR / out_name
        line_plot_by_x(summary, 'l', METHODS_LAG, 'Lag size', ylabel, path, is_size=is_size, x_order=[0, 1, 2, 3, 4], figsize=(7, 7),
                       label_fontsize=24, tick_fontsize=24, legend_fontsize=18, y_limits=y_limits, legend_loc='upper left')
        out.append(path)

    return out


def generate_mpdta_figures(source: Path) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    line_specs = [
        (8, ['sta_valid/validity_static_raw.csv', 'sta_valid/validity_static_summary.csv'], METHODS_MPDTA_STATIC, 'source_t',
         MPDTA_STATIC_SOURCE_T_ORDER, 't', 'Type I error rate', 'fig8_mpdta_sta_valid_size_by_source_t.pdf', True, SIZE_Y_LIMITS),
        (9, ['sta_power/power_static_raw.csv', 'sta_power/power_static_summary.csv'], METHODS_MPDTA_STATIC, 'source_t',
         MPDTA_STATIC_SOURCE_T_ORDER, 't', 'Power', 'fig9_mpdta_sta_power_power_by_source_t.pdf', False, RATE_Y_LIMITS),
        (10, ['lag_valid/validity_lagged_raw.csv', 'lag_valid/validity_lagged_summary.csv'], METHODS_MPDTA_LAG, 'lag',
         MPDTA_LAG_ORDER, 'Lag size', 'Type I error rate', 'fig10_mpdta_lag_valid_size_by_lag.pdf', True, SIZE_Y_LIMITS),
        (11, ['lag_power/power_lagged_raw.csv', 'lag_power/power_lagged_summary.csv'], METHODS_MPDTA_LAG, 'lag',
         MPDTA_LAG_ORDER, 'Lag size', 'Power', 'fig11_mpdta_lag_power_power_by_lag.pdf', False, RATE_Y_LIMITS),
    ]
    for _, rels, methods, x_col, x_order, xlabel, ylabel, out_name, is_size, y_limits in line_specs:
        if not source_has_any_file(source, rels):
            continue
        summary = summarize_mpdta_metric(source, rels, methods, x_col)
        path = OUT_DIR / out_name
        line_plot_by_x(
            summary, x_col, methods, xlabel, ylabel, path,
            is_size=is_size, x_order=x_order, figsize=(7, 7),
            label_fontsize=24, tick_fontsize=24, legend_fontsize=18,
            y_limits=y_limits, legend_loc='upper right',
        )
        out.append(path)

    subgroup_dirs = discover_subgroup_dirs(source, MPDTA_TAU_VALUES)
    if not subgroup_dirs:
        return out

    # Figs. 12--14: CATE profile plots for tau = 0.15, 0.20, 0.25.
    for fig_no, tau in zip([12, 13, 14], MPDTA_PROFILE_TAUS):
        key = round(float(tau), 10)
        if key not in subgroup_dirs:
            continue
        folder = subgroup_dirs[key]
        rels = [f'{folder}/{name}' for name in MPDTA_SUBGROUP_SELECTION_FILENAMES]
        if not source_has_any_file(source, rels):
            continue
        rel = first_existing_rel_path(source, rels)
        path = OUT_DIR / f'fig{fig_no}_mpdta_subgroup_selection_tau_profile_tau{tau:g}.pdf'
        plot_mpdta_subgroup_selection_profile(
            source, rel, path, rep=MPDTA_SUBGROUP_SELECTION_REP,
            figsize=(7, 7), label_fontsize=24, tick_fontsize=24, legend_fontsize=18,
        )
        out.append(path)

    # Fig. 15: selected threshold against tau. No legend.
    if all(
        source_has_any_file(source, [f'{folder}/{name}' for name in MPDTA_SUBGROUP_SELECTION_SUMMARY_FILENAMES])
        or source_has_any_file(source, [f'{folder}/{name}' for name in MPDTA_SUBGROUP_SELECTION_FILENAMES])
        for folder in subgroup_dirs.values()
    ):
        summary15 = summarize_mpdta_threshold_ci(source, subgroup_dirs)
        path15 = OUT_DIR / 'fig15_mpdta_subgroup_selection_threshold_ci_by_tau.pdf'
        plot_mpdta_threshold_ci(summary15, path15, figsize=(7, 7), label_fontsize=24, tick_fontsize=24)
        out.append(path15)

    # Figs. 16--17: subgroup inference by tau. No legends.
    if all(source_has_any_file(source, [f'{folder}/{name}' for name in MPDTA_SUBGROUP_INFERENCE_FILENAMES]) for folder in subgroup_dirs.values()):
        for fig_no, subgroup, ylabel, out_name in [
            (16, 'null_subgroup', 'Reject rate', 'fig16_mpdta_sub_power_null_subgroup_size_by_tau.pdf'),
            (17, 'effective_subgroup', 'Reject rate', 'fig17_mpdta_sub_power_effective_subgroup_power_by_tau.pdf'),
        ]:
            summary = summarize_mpdta_subgroup_taus(source, subgroup_dirs, subgroup, method='RV')
            path = OUT_DIR / out_name
            line_plot_by_x(
                summary, 'tau', METHODS_MPDTA_SUBGROUP, r'$\tau_*$', ylabel, path,
                is_size=(fig_no == 16), x_order=list(subgroup_dirs.keys()), figsize=(7, 7),
                label_fontsize=24, tick_fontsize=24, legend_fontsize=18,
                y_limits=(SIZE_Y_LIMITS if fig_no == 16 else RATE_Y_LIMITS), show_legend=False,
            )
            out.append(path)

    return out

def generate_consistency_figures(base_root: Path = DATA_ROOT) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for branch in ('nonparametric', 'parametric'):
        summary = summarize_consistency_details(read_consistency_details(base_root, branch, prefix='tau'))
        path = OUT_DIR / CONSISTENCY_BRANCH_FILENAMES[branch]
        consistency_style_line_plot(summary, x_col='N', y_col='mean_nmse', y_label='NMSE', save_path=path)
        out.append(path)
    return out



def generate_warm_start_figure(base_root: Path = DATA_ROOT) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_warm_start_summary(base_root)
    path = OUT_DIR / FIG20_WARM_START_FILENAME
    warm_start_consistency_style_line_plot(summary, save_path=path)
    return [path]

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if all(has_sample_size_sources(MAIN_ROOT, p) for p in ['sta_valid', 'sta_power', 'lag_valid', 'lag_power']):
        try:
            generated.extend(generate_original_figures(MAIN_ROOT))
        except Exception as exc:
            print(f'Skipped figures 1-7 because the original data could not be processed: {exc}')
    else:
        print('Skipped figures 1-7 because results/main sample-size folders are not all available.')

    mpdta_source = resolve_mpdta_source(DATA_ROOT)
    if mpdta_source is not None:
        try:
            generated.extend(generate_mpdta_figures(mpdta_source))
        except Exception as exc:
            print(f'Skipped MPDTA figures because the MPDTA data could not be processed: {exc}')
    else:
        print('Skipped MPDTA figures because no MPDTA source was found.')

    if all(has_consistency_source(DATA_ROOT, b, required_filename='tau_consistency_details.csv') for b in ('nonparametric', 'parametric')):
        try:
            generated.extend(generate_consistency_figures(DATA_ROOT))
        except Exception as exc:
            print(f'Skipped figures 18-19 because the consistency data could not be processed: {exc}')
    else:
        print('Skipped figures 18-19 because the consistency detail outputs are not both available.')

    if has_warm_start_source(DATA_ROOT, required_filename='tau_warm_start_summary.csv') or has_warm_start_source(DATA_ROOT, required_filename='tau_warm_start_details.csv'):
        try:
            generated.extend(generate_warm_start_figure(DATA_ROOT))
        except Exception as exc:
            print(f'Skipped figure 20 because the warm-start consistency data could not be processed: {exc}')
    else:
        print('Skipped figure 20 because results/consistency/start warm-start outputs are not available.')

    print(f'Generated {len(generated)} figure files in: {OUT_DIR.resolve()}')
    for path in generated:
        print(f' - {path.name}')


if __name__ == '__main__':
    main()
