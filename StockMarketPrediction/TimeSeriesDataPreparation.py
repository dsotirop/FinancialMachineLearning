# =============================================================================
# This script file provides fundamental computational functionality for the 
# implementation of Case Study I: Supervised Learning Models for Stock Market
# Prediction.
# =============================================================================

# =============================================================================
# In this case study, the closing value of a given stock, currency or index at 
# given time in the future can be the predicted variable. 
# We need to understand what affects each given stock, currency or index price 
# and incorporate as much information into the model. For this case study,  
# dependent and independent variables may be selected from the following list  
# of potentially correlated assets:
#
# (1) Stocks:   IBM (IBM) and Alphabet (GOOGL)
# (2) Currency: USD/JPY and GBP/USD
# (3) Indices:  S&P 500, Dow Jones, and NASDAQ
# =============================================================================

# =============================================================================
# The dataset used for this case study is extracted from https://twelvedata.com/
# You make create a free account and request the respective API KEY by accessing
# the url: https://twelvedata.com/account
# We will use the daily closing price of the last 14 years, from 2010 onward. 
# =============================================================================

# Import required Python modules.
import os
import pickle
import pandas as pd

# ============================================================================= 
#                   FUNCTIONS DEFINITION SECTION:
# =============================================================================

# =============================================================================
# This function loads the collected_data dictionary from a pickle file.
# =============================================================================
def load_collected_data(pickle_path):

    # Input Arguments:
    # pickle_path : String representing the filesystem path to the pickle file 
    #               containing the collected_data dictionary.

    # Output Arguments:
    # collected_data: Dictionary object representing the collected_data structure.
    
    # The collected_data is a dictionary object which stores the following keys:
    # (i):   stocks
    # (ii):  currencies
    # (iii): indices
    # with each element of the dictionary being a dictionary itself.
    
    # The stocks key stores the following keys:
    # (i):   MSFT (Microsoft Stock)
    # (ii):  IBM  (IBA Stock)
    # (iii): GOOGL (GOOGLE Stock)
    
    # The currencies key stores the following keys:
    # (i):  USD/JPY (USD vs JPY)
    # (ii): GBP/USD (GBP vs USD)
    
    # The indices key stores the following keys:
    # (i):   SPY (S&P 500 ETF)
    # (ii):  DIA (Dow Jones ETF)
    # (iii): QQQ (NASDAQ-100 ETF)
    
    # Exchange-Traded Fund: An ETF is an investment fund that holds a basket of
    # assets—such as stocks, bonds, or commodities—and trades on a stock exchange 
    # similar to an individual stock.

    
    # Raises FileNotFoundError:
    # If the specified pickle file is not found.
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at '{pickle_path}'")
    
    with open(pickle_path, "rb") as f:
        collected_data = pickle.load(f)
    return collected_data

# =============================================================================
# This function synchronizes all DataFrames in the collected_data dictionary
# so that they share the same DateTime index.
# =============================================================================
def synchronize_dataframes(collected_data):

    # Input Arguments:
    # - collected_data : The dictionary object which stores the downloaded data.

        
    # Output Arguements:
    # - collected_data: The same dictionary object but with each DataFrame
    #                   reindexed to the common set of dates.
    
    # Gather all DataFrames
    all_dataframes = []
    for group_data in collected_data.values():
        for df in group_data.values():
            all_dataframes.append(df)
    
    if not all_dataframes:
        # If somehow there's no data, just return original
        return collected_data
    
    # Determine the common index
    common_index = all_dataframes[0].index
    for df in all_dataframes[1:]:
        common_index = common_index.intersection(df.index)
    
    # Reindex each DataFrame
    for group, group_data in collected_data.items():
        for symbol, df in group_data.items():
            group_data[symbol] = df.reindex(common_index)

    return collected_data

# =============================================================================
# This function transforms a time-series DataFrame into a supervised learning 
# dataset.
# =============================================================================
def create_supervised_dataset(df, target_col, future_dt, past_dt, multiseries=False):
    
    
    # Input Arguments:
    # df : Original time-series DataFrame, which may have a column named 'Date',
    #      plus other columns for asset prices, indices, etc.
    # target_col : String representing the name of the column whose future value 
    #              will be predicted.
    # future_dt : Integer representing the number of future time steps upon which
    #             predictin will be performed. E.g., future_dt=1 means predict 
    #             the value at t+1 using data up to time t.
    # past_dt : Integer representing the number of past time steps that will be
    #           utilized as features. E.g., past_dt=2 means each row uses values 
    #           at [t, t-1, t-2].
    # multiseries : Boolean value indicating whether lagged features will be 
    #               generated only for the target column. If False, only the 
    #               target column will be utilized to generate lagged features.

        
    # Output Arguments:
    # supervised_df : A new DataFrame such that:
    #    - The first column is the future target (Y), 
    #      i.e. target_col shifted by -future_dt.
    #    - The next columns are the lagged features:
    #        * If multiseries=True, lagged features are created for every column 
    #          except 'Date'.
    #        * If multiseries=False, lagged features are created only for target_col.
    #    - The original 'Date' column is retained (unlagged) so you can identify 
    #      each row's date.
    #    - Rows with insufficient history or missing future values are dropped.
    
    # ------------------------------------------------------------------------
    # Extract and remove the 'Date' column (if present) so we don't generate lags for it
    # ------------------------------------------------------------------------
    date_series = df['Date'].copy() if 'Date' in df.columns else None
    
    # ------------------------------------------------------------------------
    # Create the target (Y) by shifting target_col backwards by future_dt
    # ------------------------------------------------------------------------
    y = df[target_col].shift(-future_dt)
    y.name = f"{target_col}_t+{future_dt}"  # e.g., 'MSFT_t+5'
    
    # ------------------------------------------------------------------------
    # Determine which columns to lag
    # ------------------------------------------------------------------------
    # If multiseries=True, use every column except 'Date'.
    # If multiseries=False, use only the target_col.
    if multiseries:
        columns_to_lag = [c for c in df.columns if c != 'Date']
    else:
        columns_to_lag = [target_col]  # Only the target column
    
    # ------------------------------------------------------------------------
    # Generate lagged features
    # ------------------------------------------------------------------------
    lagged_features = []
    for col in columns_to_lag:
        # Generate lagged columns for col from 0..past_dt
        for lag in range(past_dt + 1):
            col_lag_name = f"{col}_t-{lag}"
            lagged_col = df[col].shift(lag).rename(col_lag_name)
            lagged_features.append(lagged_col)
    
    # ------------------------------------------------------------------------
    # Combine target and lagged features
    # ------------------------------------------------------------------------
    supervised_df = pd.concat([y] + lagged_features, axis=1)
    
    # ------------------------------------------------------------------------
    # Drop rows with NaN (from shifting) BEFORE reintroducing 'Date'
    # ------------------------------------------------------------------------
    supervised_df.dropna(inplace=True)
    
    # ------------------------------------------------------------------------
    # Reintroduce the 'Date' column (unlagged) so each row has a date
    # ------------------------------------------------------------------------
    if date_series is not None:
        supervised_df['Date'] = date_series
    
    # ------------------------------------------------------------------------
    # Reorder columns so the final DataFrame is [Date, Target, Features...]
    # ------------------------------------------------------------------------
    if 'Date' in supervised_df.columns:
        col_order = ['Date', y.name] + [c for c in supervised_df.columns 
                                        if c not in ('Date', y.name)]
        supervised_df = supervised_df[col_order]
    
    # Reset index.
    supervised_df = supervised_df.reset_index(drop=True)
    
    return supervised_df

# =============================================================================
# This function saves a given time-series DataFrame into the designate .csv 
# file.
# =============================================================================
def save_dataframe(df, data_directory, datafile):
    
    # Input Arguments:
    # -df: Dataframe to be save into a .csv file.
    # -data_directory: String representing the location of the .csv file.
    # -data_file: String representing the name of the data file.
    
    # Check if the directory exists, if not, create it.
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Directory '{data_directory}' created.")
    else:
        print(f"Directory '{data_directory}' already exists.")

    # Construct the full data path
    data_path = os.path.join(data_directory,datafile)

    # Save the DataFrame to the file
    df.to_csv(data_path, index=False)
    print(f"DataFrame saved to '{data_path}'")

# ============================================================================= 
#                   MAIN CODE SECTION:
# =============================================================================


# ============================================================================= 
#                   LOAD DATASET:
# =============================================================================

# Set the pickle file that contains the previously downloaded data as a 
# dictionary object. 
PICKLE_FILE = './data/collected_data.pkl'

# Load dataset.
collected_data = load_collected_data(PICKLE_FILE)

# ============================================================================= 
#                   SYNCHRONIZE DATASET:
# =============================================================================
synchronize_dataframes(collected_data)

# ============================================================================= 
#                   GENERATE MAIN DATAFRAME.
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Create a dictionary to hold each "Close" series.
# -----------------------------------------------------------------------------

close_data = {}

# Step 2: Traverse the nested dictionary and extract the "Close" column from each DataFrame.
for category, sub_dict in collected_data.items():
    for ticker, df in sub_dict.items():
        # Extract the 'Close' column and store it under the key = ticker
        close_data[ticker] = df['close']

# Step 3: Build a single DataFrame that contains all the "Close" columns for all 
#         tickers.
combined_df = pd.DataFrame(close_data)

# Set the names for the columns for the complete dataset.
# • Nasdaq Composite Index often appears under the ticker symbol “IXIC”.
# • DEXUSJP corresponds to the U.S. Dollar to Japanese Yen exchange rate.
# • DEXUSUK corresponds to the U.S. Dollar to British Pound exchange rate.
column_names = ["MSFT","IBM","GOOGL","DEXUSJP","DEXUSUK","SP500","DJIA","IXIC"]
combined_df.columns = column_names

# -----------------------------------------------------------------------------
# Step 4: Convert the existing DateTimeIndex into a column named 'Date' and
#         switch to a regular integer index:
# -----------------------------------------------------------------------------

combined_df = combined_df.reset_index().rename(columns={'datetime': 'Date'})

# -----------------------------------------------------------------------------
# Step 5: Create and save the supervised learning dataframes for each column of 
#         the  combined dataframe excluding the Date series.
# -----------------------------------------------------------------------------

# Set the future dt.
future_dt = 1
# Set the past dt.
past_dt = 200
# Set the name of the data directory.
data_directory = './data'

# Loop through the various columns of the combined dataframe excluding the Date
# series.
columns_to_process = [c for c in combined_df.columns if c != 'Date']
for target_col in columns_to_process:
    
    # Create the dataset that contains only lagged versions of the target
    # regression variable.
    dataset = create_supervised_dataset(combined_df, target_col, future_dt, 
                                        past_dt)
    
    # Create the dataset that contains lagged versions of both the target and
    # the independent regression variables.
    multi_dataset = create_supervised_dataset(combined_df, target_col, future_dt, 
                                              past_dt, multiseries=True)
    
    # Save the target regression variable specific dataset to file.
    data_file = f"{target_col}_time_series_data.csv"
    save_dataframe(dataset, data_directory, data_file)
    
    # Save the multi-series target regression variable specific dataset to file.
    multi_data_file = f"{target_col}_multi_time_series_data.csv"
    save_dataframe(multi_dataset, data_directory, multi_data_file)
