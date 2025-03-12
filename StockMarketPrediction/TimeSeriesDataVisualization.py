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
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy.signal import detrend

# ============================================================================= 
#                   FUNCTIONS DEFINITION SECTION:
# =============================================================================

# =============================================================================
# This function saves the current matplotlib figure to a PNG file.
# =============================================================================
def save_figure(filename, figures_dir = "figures", fig_width=8, fig_height=6):

    # Input Arguments:
    # figures_directoty: String representing the local directory where all 
    #                    generated figures should be shaved.    
    # filename: String representing the name or path of the PNG file to save.
    
    # Check if the 'figures' folder exists; if not, create it
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Retrieve current figure and set its size (in inches)
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)

    # Construct the full path and save the figure
    full_path = os.path.join(figures_dir, filename)
    plt.savefig(full_path, dpi=100, bbox_inches='tight')
    
# =============================================================================
# This function provides a two-dimensiional plot of a dependent series object 
# y as a function of the independent series object x.
# =============================================================================
def plot_series(df, series, day_min, day_max):

    # Validate day_min and day_max within dataset range
    if not (1 <= day_min <= len(df)) or not (1 <= day_max <= len(df)):
        raise ValueError("day_min and day_max must be within the range [1, len(dataset)].")
    
    # Convert day_min and day_max to index-based slicing
    day_min_index = day_min - 1  # Convert to zero-based index
    day_max_index = day_max  # Inclusive slicing
    
    # Filter DataFrame within the given index range
    df_filtered = df.iloc[day_min_index:day_max_index]
    
    # Extract the x and y values
    x = df_filtered['Date']
    y = df_filtered[series]
    
    # Dynamically adjust figure size
    num_points = len(x)
    fig_width = max(12, num_points / 100)  
    fig_height = 6
    figsize = (fig_width, fig_height)
    
    plt.figure(figsize=figsize)
    plt.style.use('dark_background')
    
    # Plot the data
    title_str = f"{series} Prices Evolution"
    plt.plot(x, y, marker='o', color='w', linestyle='-', linewidth=1.5, 
             alpha=0.9)
    
    plt.xlabel('Date')
    plt.ylabel(series)
    plt.title(title_str)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Customize x-axis ticks dynamically
    if num_points > 25:
        tick_spacing = max(1, num_points // 25)
        plt.xticks(ticks=x[::tick_spacing], rotation=45)
    else:
        plt.xticks(rotation=45)
    
    # Save the current figure after setting the name of the image file.
    png_filename = f"{series}_time_evolution.png"
    save_figure(png_filename)
    
    plt.show()

# =============================================================================
# This function performs Fourier Transform analysis on a time series to identify 
# dominant periodicities. The dominant period is defined as:
#                             1
# Dominant Period = ------------------- 
#                    Dominant Frequency
#
# Take into consideration that the negative frequency components are discarded 
# since they constitute just the complex conjugates of the positive frequency 
# components. Moreover, they do not provide additional insights, as they mirror 
# the positive ones. More importantly, frequency f corresponds to a repeating 
# pattern every 1/f time units. Since we work with real-valued time series, we 
# focus only on the positive frequencies to extract meaningful periodicities.
# =============================================================================
def fourier_analysis(series, series_name, sampling_rate=1):
        
    # Input Parameters:
    # - series (pd.Series): Time series data indexed by DatetimeIndex.
    # - sampling_rate (float): The rate at which data points are sampled (default: 1 per day).

    # Output Parameters:
    # - dominant_period (float): The estimated dominant periodicity.
    # - freq (np.array): Array of frequency values.
    # - magnitude (np.array): Magnitude of Fourier Transform at each frequency.
    
    # Step 1: Detrend the time series (to remove long-term trends)
    series_detrended = detrend(series.dropna())  # Remove trend
    
    # Step 2: Compute the FFT (Fourier Transform)
    N = len(series_detrended)  # Number of observations
    fft_values = np.fft.fft(series_detrended)  # Compute FFT
    freq = np.fft.fftfreq(N, d=sampling_rate)  # Compute frequency bins
    
    # Step 3: Compute Magnitude Spectrum (Power Spectrum)
    magnitude = np.abs(fft_values)  # Magnitude of FFT
    
    # Step 4: Identify the Dominant Frequency
    positive_frequencies = freq[:N//2]  # Keep only positive frequencies
    positive_magnitudes = magnitude[:N//2]  # Corresponding magnitudes
    
    peak_index = np.argmax(positive_magnitudes)  # Find peak in the spectrum
    dominant_frequency = positive_frequencies[peak_index]  # Get dominant frequency
    
    # Compute the dominant period (avoid division by zero)
    dominant_period = 1 / dominant_frequency if dominant_frequency > 0 else np.nan
    
    # Step 5: Plot the Magnitude Spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(positive_frequencies, positive_magnitudes, color='cyan', lw=2)
    plt.axvline(dominant_frequency, color='red', linestyle='--', label=f"Dominant Freq: {dominant_frequency:.4f}")
    plt.title("Frequency Spectrum (Fourier Transform)")
    plt.xlabel("Frequency (cycles per unit time)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    
    # Save the current figure after setting the name of the image file.
    png_filename = f"{series_name}_fourier_spectrum.png"
    save_figure(png_filename)
    plt.show()
    
    print(f"Dominant Period: {dominant_period:.2f} time units")
    
    return dominant_period, positive_frequencies, positive_magnitudes


# ============================================================================= 
#                   MAIN CODE SECTION:
# =============================================================================

# ============================================================================= 
# Load Main Dataframe.
# =============================================================================

# Set the name of the data directory.
data_directory = './data'
data_file = 'time_series_data.csv'

# Construct the f ull data path
data_path = os.path.join(data_directory,data_file)

# Load the DataFrame from the CSV file
dataset = pd.read_csv(data_path)

# Get the time span of the given dataset in days.
days_num = len(dataset)

# Report the time span of the given dataset.
print(f"Dataset spans a time period of {days_num} days")

# ============================================================================= 
# Visualize Time Series Dataset in the Time Domain.
# =============================================================================

# Plot the first 300 days of observations for each series.
day_min, day_max = 1, 300

# Loop through the various series objects and plot the time evolution of the
# numeric data variables for the previously defined time period of days.
numeric_columns = dataset.columns[dataset.dtypes != "object"]
for series in numeric_columns:
    plot_series(dataset,series,day_min,day_max)


# ============================================================================= 
# Visualize Time Series Dataset in the Frequency Domain.
# =============================================================================

# Initialize list containers for storing the dominant period, the positive 
# frequencies and the corresponding magnitudes.
DominantPeriods = []
PositiveFrequencies = []
PositiveMagnitudes = []

# Loop through the various series objects and plot the frequency spectrogram of 
# the numeric data variables for the previously defined time period of days.
numeric_columns = dataset.columns[dataset.dtypes != "object"]
for series in numeric_columns:
    dominant_period, pos_freq, pos_mag = fourier_analysis(dataset[series], series)
    DominantPeriods.append(dominant_period)
    PositiveFrequencies.append(pos_freq)
    PositiveMagnitudes.append(pos_mag)

# The visualization involves independently understanding each attribute of the
# dataset. We will look at the scatterplot and the correlation matrix. These 
# plots give us a sense of the interdependence of the data. Correlation can be 
# calculated and displayed for each pair of the variables by creating a 
# correlation matrix. Hence, besides the relationship between independent and 
# dependent variables, it also shows the correlation among the independent  
# variables. 


# Create a copy of the dataset excluding the datetime series so that the 
# pairwise correlations may be computed.
df = dataset.drop("Date",axis=1)

# Compute the pairwise correlations amongst the selected data series.
correlation = df.corr()


# Generate the respective heam map of correlations.
plt.figure(figsize=(15,15))
plt.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
# Save and show
save_figure("correlation_matrix.png",fig_width=15, fig_height=15)
plt.show()


# Visualize the pairwise scatter plots for the variables pertaining to the 
# given regression task.
plt.figure(figsize=(15,15))
scatter_matrix(dataset,figsize=(12,12))
# Save and show
save_figure("scatter_matrix.png",fig_width=12, fig_height=12)
plt.show()
