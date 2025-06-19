# This script file provides fundamental computational functionality for 
# investigating the potential of placing a long or short position in the setting
# of algorithmic trading.

# Import required Python libraries.
import pandas as pd
import os
import matplotlib.pyplot as plt

# =============================================================================
#                      FUNCTION DEFINITION SECTION:
# =============================================================================

# This function loads the daily closing prices and volumes for a given stock 
# from a CSV file.
def load_stock_data(filename, directory):
    
    # Input Arguments:
    # filename (str): The name of the CSV file (e.g., 'AAPL.csv').
    # directory (str): The directory where the CSV file is stored.

    # Output Arguments:
    # pd.DataFrame: DataFrame with columns ['Date', 'Close', 'Volume'], sorted by date.
    
    # Construct the full path to the CSV file
    file_path = os.path.join(directory, filename)

    # This line reads a CSV file into a pandas DataFrame with specific formatting:
    # - `header=[0, 1]` tells pandas to treat the first two rows as a multi-level column header.
    #   For example, the first row might contain labels like 'Price', 'Close', 'Volume',
    #   and the second row might contain the ticker symbol, such as 'AAPL', for each column.
    #   This creates a hierarchical column structure like ('Close', 'AAPL').
    # - `index_col=0` sets the first column (typically 'Date') as the index of the DataFrame.
    #   This is important for time series data where rows are identified by dates.
    # - `parse_dates=True` automatically converts the date strings in the index column
    #   into datetime objects, enabling time-based operations and filtering.
    try:
        df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{filename}' was not found in directory '{directory}'.")

    # Simplify multi-level columns to single-level by keeping only the first level
    # and assuming the second level is the ticker (e.g., 'AAPL') repeated
    df.columns = df.columns.get_level_values(0)

    # Keep only the 'Close' and 'Volume' columns
    try:
        df_filtered = df[['Close', 'Volume']].copy()
    except KeyError:
        raise ValueError("CSV file must contain 'Close' and 'Volume' columns.")

    # Reset index and rename it to 'Date'
    df_filtered = df_filtered.reset_index().rename(columns={'index': 'Date'})

    # Sort the DataFrame by date
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
    
    return df_filtered

# This function plots the normalized daily evolution of price and volume signals.
def plot_price_volume_signals(df):
    
    # Input Arguments:
    # df (pd.DataFrame): DataFrame with columns ['Date', 'Close', 'Volume']
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    # Normalize Close and Volume
    norm_close = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    norm_volume = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())

    # Determine tick spacing
    xticks = df['Date'][::len(df) // 10]

    # Plot normalized Close
    axes[0].plot(df['Date'], norm_close, label='Normalized Close Price', linewidth=1.5)
    axes[0].set_title('Normalized Close Price Over Time')
    axes[0].set_ylabel('Normalized Price')
    axes[0].set_xticks(xticks)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Plot normalized Volume
    axes[1].plot(df['Date'], norm_volume, label='Normalized Volume', linewidth=1.5, color='orange')
    axes[1].set_title('Normalized Volume Over Time')
    axes[1].set_ylabel('Normalized Volume')
    axes[1].set_xlabel('Date')
    axes[1].set_xticks(xticks)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# This function computes elative volume scores for long, short, and neutral 
# future trading signals.
# For each date in the DataFrame:
# - Look ahead `future_time_window` days.
# - Compare future prices to current price:
# - Long Condition: P_future > P_now * (1 + alpha)
# - Short Condition: P_future < P_now * (1 - alpha)
# - Neutral Condition: All others
# - Calculate the proportion of total future volume associated with each condition.
def compute_relative_volume_signals(df, past_time_window,future_time_window, alpha):

    # Input Arguments:
    # df (pd.DataFrame): DataFrame with columns ['Date', 'Close', 'Volume'].
    # future_time_window (int): Number of future days to consider for computing 
    #                           signals.
    # past_time_window (int): Number of past days required (reserved for future 
    #                         feature design).
    # alpha (float): Threshold percentage for long/short decision 
    #                (e.g., 0.02 for 2%).

    # Output Arguments:
    # pd.DataFrame: Original DataFrame with added columns ['RelVol_Long', 
    #               'RelVol_Short', 'RelVol_Neutral'].

    results = []
    valid_range_start = past_time_window
    valid_range_end = len(df) - future_time_window

    for i in range(valid_range_start, valid_range_end):
        current_price = df.loc[i, 'Close']
        future_prices = df.loc[i+1:i+future_time_window, 'Close']
        future_volumes = df.loc[i+1:i+future_time_window, 'Volume']

        long_mask = future_prices > current_price * (1 + alpha)
        short_mask = future_prices < current_price * (1 - alpha)

        vol_total = future_volumes.sum()
        vol_long = future_volumes[long_mask].sum()
        vol_short = future_volumes[short_mask].sum()

        rel_vol_long = vol_long / vol_total if vol_total > 0 else 0.0
        rel_vol_short = vol_short / vol_total if vol_total > 0 else 0.0
        rel_vol_neutral = 1.0 - rel_vol_long - rel_vol_short if vol_total > 0 else 0.0

        results.append([
            df.loc[i, 'Date'],
            current_price,
            df.loc[i, 'Volume'],
            rel_vol_long,
            rel_vol_short,
            rel_vol_neutral
        ])

    # Create final DataFrame with computed values for valid time instances
    result_df = pd.DataFrame(
        results,
        columns=['Date', 'Close', 'Volume', 'RelVol_Long', 'RelVol_Short', 'RelVol_Neutral']
    )

    return result_df

# This function plots the relative volume signals (long, short, neutral) over 
# time 
def plot_volume_signals(df, thres):

    # Input Arguments:
    # df (pd.DataFrame): DataFrame with columns ['Date', 'RelVol_Long', 'RelVol_Short', 'RelVol_Neutral']
    # thres (float): Threshold to display as a red horizontal line
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Format x-ticks for visibility
    xticks = df['Date'][::len(df) // 10]

    # Plot each signal
    signals = ['RelVol_Long', 'RelVol_Short', 'RelVol_Neutral']
    titles = ['Relative Volume: Long Positions', 'Relative Volume: Short Positions', 'Relative Volume: Neutral Positions']

    for ax, signal, title in zip(axes, signals, titles):
        ax.plot(df['Date'], df[signal], label=signal, linewidth=1.5)
        ax.axhline(y=thres, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='Threshold')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Relative Volume')
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

# This function characterize each time instance based on relative volume thresholds.
def characterize_time_instances(df, thres):
    # Input Arguments:
    # df (pd.DataFrame): DataFrame with columns ['RelVol_Long', 'RelVol_Short', 
    #                    'RelVol_Neutral']
    # thres (float): Threshold in (1/3, 1] to determine characterizable signal

    # Output Arguments:
    # pd.DataFrame: The input DataFrame with an added 'Characterization' column
   
    def classify(row):
        if row['RelVol_Long'] > thres:
            return 'Long'
        elif row['RelVol_Short'] > thres:
            return 'Short'
        elif row['RelVol_Neutral'] > thres:
            return 'Neutral'
        else:
            return 'Non Characterizable'

    df = df.copy()
    df['Characterization'] = df.apply(classify, axis=1)
    return df

# =============================================================================
#                          MAIN CODE SECTION:
# =============================================================================

# Set the data directory.
data_directory = './data'

# Set the data file to be loaded.
data_file = 'TSLA_data.csv'

# Load the data for a given stock.
stock_df = load_stock_data(data_file, data_directory)

# Plot the normalized price and volume signals.
plot_price_volume_signals(stock_df)

# Set the value of the alpha parameter.
alpha = 0.02

# Set the sizes of the past and future time windows.
past_time_window = 100
future_time_window = 50

# Get the relative volumes of long, short and neutral positions that are actually
# happening within the given future time window for all valid time instances.
volume_signal_df = compute_relative_volume_signals(stock_df, past_time_window, 
                                                   future_time_window, 
                                                   alpha)

# Set the threshold volume value for characterizing each time instance according
# to whether its future time window is suitable for a long, short or neutral
# position.
thres = 1/3

# Plot these volume signals.
plot_volume_signals(volume_signal_df, thres)

# Get the actual characterization of each time instance.
position_labels_df = characterize_time_instances(volume_signal_df, thres)

# Show the histogram of these characterizations.
position_labels_df["Characterization"].hist()
plt.title("Distribution of Trade Signal Characterization")
plt.xlabel("Characterization")
plt.ylabel("Count")
plt.show()