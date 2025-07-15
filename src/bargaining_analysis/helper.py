# Precompile the pattern to find any \roever{var}{old_value}
import re
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def fix_pandas_append_error():
          #Fix pandas append error
    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, sort=False):
            return pd.concat([self, other],
                            ignore_index=ignore_index,
                            sort=sort)
        pd.DataFrame.append = _append


def inject_values(tex_path: Path, **variables):
    """
    Reads a .tex file, finds all \roever{var}{...} placeholders,
    replaces the ... with the provided variables[var], and writes back.
    
    Parameters:
    - tex_path: pathlib.Path to the .tex file
    - variables: kwargs mapping var names to their replacement values
    
    Raises:
    - KeyError: if a var placeholder isn't found or if any remain afterward
    """
    path = Path(tex_path)
    content = path.read_text(encoding='utf-8')
    ROEVER_PATTERN = re.compile(r'\\roever\{(?P<var>\w+)\}\{[^}]*\}')
    
    # Replace each variable's placeholder via regex substitution
    for var, val in variables.items():
        pattern = re.compile(rf'\\roever\{{{var}\}}\{{[^}}]*\}}')
        replacement = rf'\\roever{{{var}}}{{{val}}}'
        content, count = pattern.subn(replacement, content)
        if count == 0:
            raise KeyError(f"No placeholder \\roever{{{var}}}{{...}} found in {tex_path}")
    
    # Check for any unreplaced placeholders
    #leftovers = ROEVER_PATTERN.findall(content)
    #if leftovers:
    #    raise KeyError(f"Unreplaced placeholders remain for variables: {set(leftovers)}")
    
    # Write the updated content back to the file
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
    