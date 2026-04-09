# How to Code Regression Tables


For regression tables, use the pystout package.
Here is an example of how a function should look like. 
Read in the cleaned dataset, the readable labels (it is a dictionary given in config.py) and the output path for the table. 

```python
from pystout import pystout
from src.bargaining_analysis.helper import fix_pandas_append_error()

def run_mechanism_regressions_w3(
    df, varlabels_regression, output_path_tex
):

    atq_w3 = smf.ols(
        formula="atq_score_std_w3 ~ sonia_treatment + atq_score_std_w2 + therapy_history_w1 + efficacy_human_therapy_w2",
        data=df,
    ).fit(cov_type="HC3")

    beh_act_w3 = smf.ols(
        formula="beh_act_score_std_w3 ~ sonia_treatment + beh_act_score_std_w2 + therapy_history_w1 + efficacy_human_therapy_w2",
        data=df,
    ).fit(cov_type="HC3")

    bias_w3 = smf.ols(
        formula="bias_score_std_w3 ~ sonia_treatment + bias_score_std_w2 + therapy_history_w1 + efficacy_human_therapy_w2",
        data=df,
    ).fit(cov_type="HC3")

    fix_pandas_append_error()

    pystout(
        models=[atq_w3, beh_act_w3, bias_w3],
        endog_names=[
            r" \shortstack{ Automatic Thoughts \\ (Standardized)}",
            r" \shortstack{ Behavioral Activation \\ (Standardized)}",
            r" \shortstack{ Cognitive Bias \\ (Standardized)}",
        ],
        file=output_path_tex,
        digits=2,
        stars={0.1: "*", 0.05: "**", 0.01: "***"},
        varlabels=varlabels_regression,
        modstat={"nobs": "Obs", "rsquared_adj": r"Adj. R\sym{2}"},
        addrows={
            "Baseline Controls": [
                "\\checkmark",
                "\\checkmark",
                "\\checkmark",
            ]
        },
        exogvars=["sonia_treatment"],
    )

```

