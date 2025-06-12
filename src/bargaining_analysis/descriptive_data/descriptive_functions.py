from src.bargaining_analysis.helper import finalize_plot
import pandas as pd
import matplotlib.pyplot as plt


def plot_time_preference_switching_points(df):
    """
    Takes a DataFrame and returns a bar plot counting the occurrences of values 1 to 7
    in the 'time_preference_switching_points' column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'time_preference_switching_points' column.

    Returns:
    matplotlib.axes._subplots.AxesSubplot: The Axes object with the plot.
    """
    df_round33 = df[df['round'] == 33]
    
    # Count occurrences of each value from 1 to 7
    counts = df_round33["time_preference_switching_points"].value_counts().reindex(range(1, 8), fill_value=0)
    
    # Create the bar plot
    ax = counts.plot(kind='bar')
    ax.set_xlabel('Switch to receiving money later at interest rate')
    ax.set_ylabel('Count')
    ax.set_xticks(range(0, 7))
    ax.set_xticklabels([r'$\le 5\%$', r'$10\%$', r'$15\%$', r'$20\%$', r'$25\%$', r'$30\%$', r'$>30\%$'])
    ax.tick_params(axis='x', rotation=45)  # Angle the x ticks
    finalize_plot(ax)
    return ax


def plot_ultimatum_offer_histogram(df):
    """
    Plots a histogram of ultimatum offers for round 33.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    None
    """
    ax = df[df["round"] == 33]["ultimatum_offer"].hist()
    plt.xlabel('Money to Keep in Euros')
    plt.ylabel('Frequency')
    plt.title('Histogram of Ultimatum Offers in Round 33')
    finalize_plot()
    return ax


def plot_risk_elicitation_choices(df):
    risk_labels = [
        r"80\% chance of winning €2",
        r"70\% chance of winning €3",
        r"60\% chance of winning €4",
        r"50\% chance of winning €5",
        r"40\% chance of winning €6",
        r"30\% chance of winning €7"
    ]
    risk_counts = (
        df["risk_elicitation_choice"]
        .value_counts()
        .reindex(range(1, 7), fill_value=0)
    )

    # Create figure & axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw horizontal bars on that Axes
    ax.barh(risk_labels, risk_counts.values)

    # Now set labels and title on the Axes
    ax.set_xlabel("Count")
    ax.set_ylabel("Risk Elicitation Choice")
    ax.set_title("Counts of Risk Elicitation Choices")

    # Tidy up
    fig.tight_layout()

    # If you have a finalize_plot() utility, pass the figure to it
    finalize_plot(ax=ax)





