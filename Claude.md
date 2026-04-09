# Bargaining Experiment Analysis

In this project, we analyze the results of a bargaining experiment. 
The experiment has a 2x2 design: (1) The buyer knows the true value of the good for the seller (asymmetric uncertainty) or not (symmetric uncertainty, which means both do not know the true value for the other player). (2) Either there are time costs of 5 cts/second or not. The treatments are indicated in the dataset in the column "treatment", which has one of the following values:

- "T4": Buyer only uncertainty with transaction costs
- "T3": Buyer only uncertainty without transaction costs
- "T2": Symmetric uncertainty with transaction costs
- "T1": Symmetric uncertainty without transaction costs

## Structure of the Code

- `src/bargaining_analysis/`: This is the main source directory for the code. It contains the code for data cleaning, plotting, regression tables etc.
- `src/bargaining_analysis/clean_data/`: Here, I clean the raw data. You do not need to touch this, this is already finalized.
- `src/bargaining_analysis/main_results/`: Here, write your analysis code. 

## Common Commands

```bash
# Install environment
conda activate bargaining_analysis
```


## Further Information

The following files are available for further information: 

- `claude_assets/data_description.md`: Descriptions of the dataset
- ``claude_assets/plotting.md``: Instructions for how to code plots 
- ``claude_assets/regression_tables.md``: Instructions for how to code regression tables


## Instructions for Coding 

If you make an analysis, please write a separate file. 
The analysis should be fully in functions which take the cleaned data as an input (`BLD / "data" / "merged_data_full_excluded.csv"`) and return the output (e.g. a figure object, a regression table etc.). If it is a figure, save in OVERLEAF_FIGURES, if it is a table, save in OVERLEAF_TABLES. 

On the bottom of the file, then run the analysis function and saves the output. Example:

```python

def plot_data(
    df,
    figsize=(9, 5),
):
    ... function implementation ...
    return fig

if __name__ == "__main__":
    df = pd.read_parquet(BLD / "data" / "merged_data_full_excluded.csv")
    fig = plot_data(df=df)
    fig.savefig(OVERLEAF_FIGURES / "data_plot.pdf")    
```

If you make an analysis where you study the timing (i.e. offer times), exclude observations which are in the first four sessions (the dates are: 2025-06-11, 2025-06-13, 2025-06-16, 2025-06-17; you can extract them from `experiment_start_time` column in the dataset, which is in Unix time format, so you need to convert it to datetime first). This is because there were some technical issues in the first four sessions which led to very long offer times.

### Key Configuration

- `src/bargaining_analysis/config.py`: Central path definitions (`SRC`:source directory of code, `BLD`: build directory for outputs, `OVERLEAF_FIGURES`: path to save figures, `OVERLEAF_TABLES`: path to save tables, `OVERLEAF_ROOT`: path to main.tex for injecting values into the main text). 
- The cleaned dataset you should use for all analyses is located at `BLD / "data" / "merged_data_full_excluded.csv"`.


### Numbers Quoted in the Main Text

For numbers quoted in the main text, use the helper function `inject_values()` from `src/mi_commitment/helper.py` to inject the values into the main.tex file.The workflow is: 1) write a function which calculates the value you want to quote from the cleaned data and returns a dictionary with a key describing the value and the value itself, 2) call this function which and call `inject_values()` to inject the value into main.tex. Example:

```python

def calculate_avg_improv_therapy_w2(df):
    avg_improv_therapy_w2 = df["efficacy_human_therapy_w2"].mean()
    return {"avg_improv_therapy_w2": avg_improv_therapy_w2}

if __name__ == "__main__":
    df = pd.read_parquet(BLD / "data" / "merged_data_full_excluded.csv")
    values_dict = calculate_avg_improv_therapy_w2(df=df)
    inject_values(
        OVERLEAF_ROOT / "main.tex", avg_improv_therapy_w2=round(avg_improv_therapy_w2, 2)
    )
```
