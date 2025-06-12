# Precompile the pattern to find any \roever{var}{old_value}
import re
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


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
        style="white",           # consistent with file_context_0
        palette="deep",          # or your own list of colors
        font="serif",            # consistent with file_context_0
        font_scale=1.1
    )

    # tweak matplotlib rcParams you care about
    plt.rcParams.update({
        "text.usetex":       True,  # consistent with file_context_0
        "axes.titlesize":    16,
        "axes.labelsize":    14,
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
    ax.figure.tight_layout()
