# =============================================================================
# This script file provides fundamental computational functionality for the 
# implementation of Case Study I: Supervised Learning Models for Stock Market
# Prediction.
# =============================================================================

# =============================================================================
# In this case study, the weekly return of Microsoft stock is the predicted 
# variable. We need to understand what affects Microsoft stock price and 
# incorporate as much information into the model. For this case study, other 
# than the historical data of Microsoft, the independent variables used are the 
# following potentially correlated assets:
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

# Import required Python modules.
import os
import pickle
import numpy as np
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

# The predicted variable Y is the weekly return of Microsoft (MSFT). The number 
# of trading days in a week is assumed to be five, and we compute the return 
# using five trading days.

# Initially set the time period based on which the return values will be 
# computed
return_period = 5

# Set the dependent variable to be predicted as the daily closing value of the 
# Microsoft stock. Mind that the [["close"]] slicing returns a dataframe object
# and not a series object.
Y = collected_data["stocks"]["MSFT"][["close"]]

# Convert the acquired values in log scale, compute the correponding weekly 
# returns and shift the available time series data towards the past so that 
# the last return_period values are equal to NaN.
Y = np.log(Y).diff(return_period).shift(-return_period)

# The variables used as independent variables are lagged five-day return of 
# stocks (IBΜ and GOOGLE), currencies (USD/JPY) and GBP/USD), and indices 
# (S&P 500, Dow Jones, and NASDAQ), along with lagged 5-day, 15-day, 30-day and 
# 60-day return of MSFT.

# Retrieve the IBM and GOOGLE closing stock values for the same time
# period and combine them into a dataframe object for further pre-processing.
X_ibm = collected_data["stocks"]["IBM"]["close"]
X_google = collected_data["stocks"]["GOOGL"]["close"]
X1 = pd.concat([X_ibm,X_google],axis=1)

# Convert the acquired values in log scale and compute the 
# corresponding weekly returns.
X1 = np.log(X1).diff(return_period)

# Retrieve the closing values for the JPY/USD and GBP/USD currencies for the 
# same time period and combine them into a dataframe object for further 
# pre-processing.
X_usdjpy = collected_data["currencies"]["USD/JPY"]["close"]
X_gbpusd = collected_data["currencies"]["GBP/USD"]["close"]
X2 = pd.concat([X_usdjpy,X_gbpusd],axis=1)

# Convert the acquired values in log scale and compute the corresponding weekly
# returns.
X2 = np.log(X2).diff(return_period)

# Retrieve the closing values for the SP500, DOWJONES and NASDAQ indices for the 
# same time period and combine them into a dataframe object for further 
# pre-processing.
X_sp500 = collected_data["indices"]["SPY"]["close"]
X_dowjones = collected_data["indices"]["DIA"]["close"]
X_nasdaq = collected_data["indices"]["QQQ"]["close"]
X3 = pd.concat([X_sp500,X_dowjones,X_nasdaq],axis=1)

# Convert the acquired values in log scale and compute the corresponding weekly
# returns.
X3 = np.log(X3).diff(return_period)

# Generate the lagged returns of the MSFT stock.
Xo = collected_data["stocks"]["MSFT"]["close"]
Xo = np.log(Xo)
X4 = [Xo.diff(i*return_period) for i in [1,3,6,12]]
# Concatenate the lagged MSFT stock closing values into a dataframe.
X4 = pd.concat(X4,axis=1)

# Concatenate all independent variables into a unified dataframe.
X = pd.concat([X1,X2,X3,X4],axis=1)

# Generate the complete dataframe by concatenating the dependent variables with
# the independent ones into a single dataframe by excluding the NaN values.
dataset = pd.concat([Y,X],axis=1).dropna()

# Set the names for the columns for the complete dataset.
# Nasdaq Composite Index often appears under the ticker symbol “IXIC”.
# • DEXUSJP corresponds to the U.S. Dollar to Japanese Yen exchange rate.
# • DEXUSUK corresponds to the U.S. Dollar to British Pound exchange rate.
column_names = ["MSFT","IBM","GOOGL","DEXUSJP","DEXUSUK",
"SP500","DJIA","IXIC","MSFT_DT","MSFT_3DT","MSFT_6DT","MSFT_12DT"]
dataset.columns = column_names

# Convert the existing DateTimeIndex into a column named 'Date' and
# switch to a regular integer index:
dataset = dataset.reset_index().rename(columns={'datetime': 'Date'})

# Take into consideration that the first series object will be the target
# variable while the rest series objects will be the independent variables 
# upon which the target variable will be predicted.

# ============================================================================= 
#                   SAVE TIME SERIES DATASET:
# ============================================================================= 

# ============================================================================= 
# PHASE III: Save Main Dataframe.
# =============================================================================

# Set the name of the data directory.
data_directory = './data'
data_file = 'time_series_data.csv'

# Check if the directory exists, if not, create it.
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
    print(f"Directory '{data_directory}' created.")
else:
    print(f"Directory '{data_directory}' already exists.")

# Construct the full data path
data_path = os.path.join(data_directory,data_file)

# Save the DataFrame to the file
dataset.to_csv(data_path, index=False)
print(f"DataFrame saved to '{data_path}'")
