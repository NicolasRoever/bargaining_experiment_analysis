# Precompile the pattern to find any \roever{var}{old_value}
import re
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

def fix_pandas_append_error():
          #Fix pandas append error
    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, sort=False):
            return pd.concat([self, other],
                            ignore_index=ignore_index,
                            sort=sort)
        pd.DataFrame.append = _append


def _stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.1:   return "*"
    return ""

def _ols_row(label, y_col, data, groups_col, h0_slope):
        mod = smf.ols(f"{y_col} ~ valuation", data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data[groups_col]}
        )
        sl = mod.params["valuation"]
        se = mod.bse["valuation"]
        ic = mod.params["Intercept"]
        p  = mod.pvalues["valuation"]
        # One-sided t-test: slope < h0_slope
        t_vs_h0 = (sl - h0_slope) / se
        p_vs_h0 = stats.t.cdf(t_vs_h0, df=mod.df_resid)   # p(slope < h0)
        print(f"  {label}")
        print(f"    Intercept = {ic:+.3f}")
        print(f"    Slope     = {sl:+.4f}   SE = {se:.4f}   p(≠0) = {p:.4f} {_stars(p)}")
        print(f"    H0: slope = {h0_slope}  →  t = {t_vs_h0:+.3f},  "
              f"p(slope < {h0_slope}) = {p_vs_h0:.4f} {_stars(p_vs_h0)}")
        print(f"    R² = {mod.rsquared:.4f}")
        return sl, se, p, p_vs_h0

def _clustered_mean_test(series, groups, h0_mean=0.0):
    """OLS series ~ 1, clustered SEs. Returns (mean, se, t, p)."""
    d = pd.DataFrame({"y": series, "g": groups}).dropna()
    mod = smf.ols("y ~ 1", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["g"]}
    )
    coef = mod.params["Intercept"]
    se   = mod.bse["Intercept"]
    t    = (coef - h0_mean) / se
    p    = 2 * (1 - stats.t.cdf(abs(t), df=mod.df_resid))
    return coef, se, t, p

def inject_values(tex_path: Path, **variables):
    """
    Reads a .tex file, finds all \roever{var}{...} placeholders,
    replaces the ... with the provided variables[var], and writes back.

    If a provided value is numeric and has more than 2 decimal places,
    it is rounded to 2 decimal places.
    """
    path = Path(tex_path)
    content = path.read_text(encoding='utf-8')

    def format_value(val):
        if isinstance(val, float):
            s = str(val)
            if "." in s and len(s.split(".")[1]) > 2:
                return str(round(val, 2))
            return s
        return str(val)

    for var, val in variables.items():
        pattern = re.compile(rf'\\roever\{{{var}\}}\{{[^}}]*\}}')
        formatted_val = format_value(val)
        replacement = rf'\\roever{{{var}}}{{{formatted_val}}}'
        content, count = pattern.subn(replacement, content)

    path.write_text(content, encoding='utf-8')



def set_plot_theme():
    # base seaborn theme & palette
    sns.set_theme(
        style="white",           # consistent with file_context_0        # or your own list of colors
        font="serif",            # consistent with file_context_0
        font_scale=1.4           # Increased font scale for larger text
    )

    palette = ["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"]
    sns.set_palette(palette = palette, n_colors=5)

    # tweak matplotlib rcParams you care about
    plt.rcParams.update({
        "text.usetex":       True,  # consistent with file_context_0
        "axes.titlesize":    18,    # Increased title size
        "axes.labelsize":    16,    # Increased label size
        "legend.frameon":    False,
        "figure.figsize":    (8, 5),
        "lines.linewidth":   2,
        "lines.markersize":  6,
        "axes.grid":         False, # Disable grid
        # …any other defaults…
    })

def finalize_plot(ax=None):
    if ax is None:
        ax = plt.gca()
    sns.despine(ax=ax)
    
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("white")
    
    ax.figure.tight_layout()

def validate_column_range(
    col: pd.Series,
    min_val: float = -1000,
    max_val: float = 1000
) -> None:
    """
    Checks that all non-NaN values in the Series `col` lie between `min_val` and `max_val` (inclusive).
    NaNs are ignored. Raises a ValueError if any non-NaN value is outside this range.

    Parameters
    ----------
    col : pd.Series
        The column to validate.
    min_val : float
        Minimum allowable value (default: -1000).
    max_val : float
        Maximum allowable value (default: 1000).

    Raises
    ------
    ValueError
        If any non-NaN value in `col` is < min_val or > max_val.
    """
    # Mask of entries that are non-NaN but out of range
    out_of_range_mask = col.notna() & ~col.between(min_val, max_val, inclusive="both")

    if out_of_range_mask.any():
        bad_vals = col.loc[out_of_range_mask].unique()
        sample = bad_vals[:10].tolist()
        ellipsis = "…" if len(bad_vals) > 10 else ""
        raise ValueError(
            f"Column contains values outside [{min_val}, {max_val}]: {sample}{ellipsis}"
        )
    