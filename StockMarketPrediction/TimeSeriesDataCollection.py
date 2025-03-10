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
# =============================================================================

# Import required Python modules.
import requests
import datetime
import pandas as pd
import time
import pickle
import os

# Set the Twelve Data API Key
TWELVE_DATA_API_KEY = "05b22eb6c3c449379f45ac6571f5a4b3"

# ============================================================================= 
#                   FUNCTIONS DEFINITION SECTION:
# =============================================================================

# =============================================================================
# This function ensures taht we don't exceed 'max_requests' within 'window_sec' 
# seconds. In particular, it undertakes the following tasks:
#   i) Prunes timestamps older than 'window_sec' from the current time.
#  ii) If still at or beyond the limit, raises a RuntimeError.
# iii) Returns the pruned list plus a summary (used, left).
# =============================================================================
def check_rate_limit(request_timestamps, max_requests=8, window_sec=60):
    # Input Arguments:
    # - request_timestamps : A list of UNIX timestamps (time.time()) for prior 
    #                        requests.
    # - max_requests : The maximum number of allowed requests within the time 
    #                  window, by default 8.
    # - window_sec : The size of the rolling time window in seconds, by default 
    #                60.

    # Output Arguments:
    # - pruned_timestamps : The updated list of timestamps, removing those older 
    #                       than 'window_sec' seconds.
    # - used : How many requests have been made in the last 'window_sec' seconds
    #          (including the one we are about to make).
    # -left : How many requests remain before reaching 'max_requests' in
    #         the current rolling window.

    # Raises RuntimeError:
    # If adding another request would exceed the rate limit. The
    # exception message includes how long you need to wait before
    # trying again.
    
    # Get the current time.
    now = time.time()
    # Keep only timestamps within the last 'window_sec'
    pruned_timestamps = [t for t in request_timestamps if (now - t) < window_sec]
    used = len(pruned_timestamps)  # used calls in the last 60 seconds
    left = max_requests - used      # how many calls remain in this window

    if used >= max_requests:
        oldest_in_window = pruned_timestamps[0]
        wait_time = int(window_sec - (now - oldest_in_window)) + 1
        raise RuntimeError(
            f"Rate limit exceeded: {max_requests} requests in the last {window_sec}s.\n"
            f"Please wait ~{wait_time} seconds before trying again."
        )

    return pruned_timestamps, used, left

# =============================================================================
# This function fetches historical data from the Twelve Data time_series endpoint
# for a given symbol and date range, returning a pandas DataFrame.
# =============================================================================
def fetch_twelvedata(symbol, start_dt, end_dt, interval='1day', 
                     apikey=TWELVE_DATA_API_KEY):
    # Input Arguments:
    # - symbol : String representing the market symbol recognized by Twelve Data 
    #            (e.g. "MSFT", "USD/JPY").
    # - start_dt : datetime.datetime object representing the starting date for 
    #              fetching historical data.
    # - end_dt : datetime.datetime object representing the ending date for 
    #            fetching historical data.
    # - interval : String representing the time interval for the data, such as 
    #             '1day', '1week', '1month'.
    # - apikey : String representing the Twelve Data API key.

    # Output Arguements:
    # df : A pandas.DataFrame object containing columns like 'open', 'high', 
    #      'low', 'close', 'volume', indexed by 'datetime' in ascending order.

    # Raises ValueError:
    # If the API response indicates an error status, or if the expected data 
    # fields (e.g., "values", "datetime") are missing.
    
    base_url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "apikey": apikey,
        "outputsize": 5000
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()

    # Check for error
    if "status" in data and data["status"] == "error":
        msg = data.get('message', 'Unknown error')
        raise ValueError(f"Error from Twelve Data API for '{symbol}': {msg}")

    if "values" not in data:
        raise ValueError(f"No 'values' found for symbol '{symbol}'. Response: {data}")

    df = pd.DataFrame(data["values"])

    # Convert numeric columns if they exist
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Ensure 'datetime' column is present
    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' in response for symbol '{symbol}'.")
    
    # Convert to proper DateTimeIndex
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)

    return df

# ============================================================================= 
# This function Downloads datasets from Twelve Data for the symbol groups specified
# in 'td_symbol_groups' within the global configuration. Checks and enforces a 
# rolling rate limit of 8 requests/minute, printing usage  info for each call. 
# If the rate limit is exceeded, raises RuntimeError.
# =============================================================================
def download_dataset(td_symbol_groups, apikey=TWELVE_DATA_API_KEY):
    # Raises RuntimeError:
    # If the rolling rate limit is exceeded, this function re-raises the error 
    # after printing a message. This stops further downloads
    # until the user waits.
    
    try:
        collected_data = {}
        request_timestamps = []  # track timestamps for rate-limiting

        for group_name, symbols in td_symbol_groups.items():
            print(f"\n--- Downloading {group_name} ---")
            group_data = {}
            
            for symbol in symbols:
                # Check rate limit first
                request_timestamps, used, left = check_rate_limit(
                request_timestamps, max_requests=8, window_sec=60)
                print(f"Request usage in last 60s: used={used}, left={left}")

                # Fetch data
                try:
                    df = fetch_twelvedata(symbol, start_date, end_date, interval, apikey)
                    group_data[symbol] = df
                    print(f"Downloaded {symbol}, rows={len(df)}")
                except Exception as ex:
                    print(f"Error downloading {symbol}: {ex}")

                # Record the time of this request
                request_timestamps.append(time.time())

            collected_data[group_name] = group_data

        print("\nAll downloads finished.\n")
        return collected_data
    except RuntimeError as e:
        print(f"\n** Rate Limit Error **\n{e}")

# ============================================================================= 
#                   MAIN CODE SECTION:
# =============================================================================


# ============================================================================= 
#                  DOWNLOAD DATASET:
# =============================================================================

# Set the time range for the data to be downloaded.
start_date = datetime.datetime(2010, 1, 1)
end_date   = datetime.datetime(2024, 1, 1)
interval   = '1day'  # daily data

# Define the symbol groups to be downloaded directly from Twelve Data.
td_symbol_groups = {
    "stocks": [
        "MSFT",    # Microsoft
        "IBM",     # IBM
        "GOOGL"    # Alphabet (Google)
    ],
    "currencies": [
        "USD/JPY", # USD vs JPY
        "GBP/USD"  # GBP vs USD
    ],
    "indices": [
        "SPY",     # S&P 500 ETF
        "DIA",     # Dow Jones ETF
        "QQQ"      # NASDAQ-100 ETF
    ]
}

# Perform the actual data downloading.
collected_data = download_dataset(td_symbol_groups, apikey=TWELVE_DATA_API_KEY)

# ============================================================================= 
#                   SAVE DATASET:
# ============================================================================= 

# Set the directory for saving the downloaded data.
data_directory = './data'
# Set the data file for saving the downloaded data.
data_file = 'collected_data.pkl'
# Create the data storage directory if it does not exist.
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
save_path = os.path.join(data_directory, data_file)
# Save data to the previously generated file.
with open(save_path, "wb") as f:
    pickle.dump(collected_data, f)
print(f"Data successfully saved to '{save_path}'")
