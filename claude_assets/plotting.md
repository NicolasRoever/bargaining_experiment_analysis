# Instructions for Plotting

For plots, use the helper functions in `src/mi_commitment/helper.py`. 
Call
`set_plot_theme()` at the start of your plotting function to ensure consistent styling,
and call `finalize_plot(ax=ax)` at the end before saving.

The function should take as argument the cleaned data, and return the figure object. You can access the primary, secondary color etc. like this: color=sns.``sns.color_palette()[0]``,
Here is an example of how a plotting function should look like:

```python
def plot_em_support_importance_violins(
    df,
    figsize=(9, 5),
):
    """
    Violin plots of emotional support importance items split by research arm.
    Categories are sorted by mean (lowest left, highest right), and the mean
    is annotated above each violin.
    """
    set_plot_theme()

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette = {arm: colors[i % 2] for i, arm in enumerate(arms)}

    sns.violinplot(
        data=long,
        x="item",
        y="score",
        order=order,
        ax=ax,
        hue=arm_col,
        split=len(arms) == 2,
        palette=palette,
        cut=0,
        inner=None,
    )

     .... 

    finalize_plot(ax=ax)

    plt.close()
    return fig

```