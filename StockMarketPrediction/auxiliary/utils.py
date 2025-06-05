# This is a Python library containing the implementation for various auxiliary
# functions.

# Import required Python libraries.
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import math
import seaborn as sns
# ============================================================================= 
#                   HELPER FUNCTIONS DEFINITION SECTION:
# =============================================================================

# Function to generate lagged features from the time series
def create_lagged_features(series, n_lags):

    # Input Arguments:
    # - series: a pandas.Series object representing the input time series to  
    #           create lagged features from.
    # - n_lags: an int representing the number of lags to generate.
    
    # Output Arguments:
    # - lagged_data: a numpy.ndarray where each row is a set of lagged values 
    #                for a given time point.
    
    lagged_data = []
    for i in range(n_lags, len(series)):
        lagged_data.append(series[i - n_lags:i].values)
    lagged_data = np.array(lagged_data)
    return lagged_data

# Function to characterize a given cluster based on its statistical properties.
def characterize_cluster(mean, variance, skewness, kurtosis, thresholds):

    # Input Arguments:
    # - mean: float represening the mean of the cluster.
    # - variance: float representing the variance of the cluster.
    # - skewness: float representing the skewness of the cluster.
    # - kurtosis: float representing the kurtosis of the cluster.
    # - thresholds: dictionary containing customizable thresholds for mean, 
    #               variance, skewness, and kurtosis.
    
    # Output Arguments:
    # - cluster_type: A string describing the cluster type.
    
    # Get customizable thresholds from the dictionary
    var_thresh = thresholds.get('variance', 0.1)
    mean_thresh = thresholds.get('mean', 0.1)
    skew_thresh = thresholds.get('skewness', 1)
    kurt_thresh = thresholds.get('kurtosis', 3)
    
    # Characterize cluster based on thresholds
    if variance < var_thresh:
        cluster_type = "Stable"
    elif mean > mean_thresh and skewness < 0:
        cluster_type = "Growth"
    elif mean < -mean_thresh and skewness > 0:
        cluster_type  = "Decline"
    elif kurtosis > kurt_thresh:
        cluster_type =  "Volatile"
    elif abs(skewness) > skew_thresh:
        cluster_type = "Skewed"
    else:
        cluster_type = "Normal"
    
    # Return the identfied type of cluster.
    return cluster_type

# Function to perform clustering using DTW and generate the cluster information
def perform_clustering(series, dates, n_lags, n_clusters=3, n_init=200,
                       thresholds=None):

    
    # Input Arguments:
    # - series: a pandas.Series representing the time series to cluster.
    # - dates: a pandas.Series representing the corresponding dates for the time 
    #          series data points.
    # - n_lags: an integer value representing the number of lags to generate as 
    #           features.
    # - n_clusters: an integer value representing the number of clusters to create.
    # - n_init: an integer value indicating the number of re-runs for each instance
    #           of the k-means algorithm.
    # - thresholds: an optional dictionary object containing customizable thresholds 
    #               for mean, variance, skewness, and kurtosis to characterize each 
    #               cluster.
    
    # Output Arguments:
    # - clusters: A pandas.DataFrame containing the cluster assignments, 
    #             start/end times, basic statistics of each cluster,
    #             and a textual characterization of the cluster.
   
    # Use default thresholds if none provided
    if thresholds is None:
        thresholds = {'variance': 0.1, 'mean': 0.1, 'skewness': 1, 'kurtosis': 3}
    
    # Generate lagged features
    X = create_lagged_features(series, n_lags)
    
    # Make sure the series object is defined on the same time range.
    series = series[n_lags:]

    # Normalize the data (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute the pairwise distance matrix using Pearson correlation
    dist_matrix = pairwise_distances(X_scaled, metric='correlation')

    # Perform KMeans clustering using the Pearson correlation distance matrix
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42,
                    verbose=1)
    cluster_labels = kmeans.fit_predict(dist_matrix)

    # Create a DataFrame to hold the clustering results
    clusters_df = pd.DataFrame({
        'Date': dates[n_lags:],  # dates corresponding to the time steps
        'Cluster': cluster_labels
    })

    # Adding basic statistics for each cluster
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_data = series[cluster_labels == cluster_id]
        start_date = clusters_df[clusters_df['Cluster'] == cluster_id]['Date'].iloc[0]
        end_date = clusters_df[clusters_df['Cluster'] == cluster_id]['Date'].iloc[-1]
        mean = np.mean(cluster_data)
        variance = np.var(cluster_data)
        skewness = stats.skew(cluster_data)
        kurtosis = stats.kurtosis(cluster_data)
        # Add cluster description based on stats and adjustable thresholds
        cluster_description = characterize_cluster(mean, variance, skewness, kurtosis, thresholds)
        cluster_stats.append({
            'Cluster': cluster_id,
            'Start Date': start_date,
            'End Date': end_date,
            'Mean': mean,
            'Variance': variance,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Characterization': cluster_description
        })
    
    cluster_info_df = pd.DataFrame(cluster_stats)
    
    # Use the dark theme for the plot
    plt.style.use('dark_background')

    # Visualization of the clusters in separate windows with grid
    for cluster_id in range(n_clusters):
        plt.figure(figsize=(12, 6))  # Create a new figure for each cluster
        cluster_data = series[cluster_labels == cluster_id]
        cluster_dates = clusters_df[clusters_df['Cluster'] == cluster_id]['Date']
    
        plt.plot(cluster_dates, cluster_data, label=f"Cluster {cluster_id}",
                 color='red')
        plt.title(f"Time Series Clustering - Cluster {cluster_id} (Pearson Correlation)")
        plt.xlabel("Date")
        plt.ylabel("Value")
    
        # Dynamically calculate the step size based on the number of observations
        num_dates = len(cluster_dates)  # Total number of dates/observations in this cluster
        max_ticks = 10  # Maximum number of x-ticks you want to display

        # Calculate the step size based on the number of observations and max_ticks
        step_size = max(1, num_dates // max_ticks)  # Ensure step size is at least 1

        # Set the x-ticks to every `step_size`-th date
        date_ticks = cluster_dates[::step_size]
        plt.xticks(date_ticks, rotation=45)  # Rotate x-tick labels by 45 degrees

        # Display grid and set grid style
        plt.grid(True, linestyle='--', linewidth=0.5, color='white')  # White grid on dark background
    
        # Add legend
        plt.legend()
    
        # Show the plot for the current cluster
        plt.show()
    
    return cluster_info_df, clusters_df

# Define a function to get the correct training environemnt for the model.
def get_execution_device():
    # Set the existence status of a mps GPU.
    if hasattr(torch.backends,"mps"):
        is_mps = torch.backends.mps.is_available()
    else:
        is_mps = False
    # Set the existence status of a cuda GPU.
    is_cuda = torch.cuda.is_available()
    # Check the existence status of a mps GPU to be used during training.
    if is_mps:
        device = torch.device("mps")
        print("MPS GPU is available!")
        print(70*"=")
    # Check the existence of a cuda GPU to be used during training.
    elif is_cuda:
        device = torch.device("cuda")
        print("CUDA GPU is available!")
        print(70*"=")
    # Otherwise, a CPU device will be used instead.
    else:
        device = torch.device("cpu")
        print("GPU is not available, CPU will be used instead!")
        print(70*"=")
    return device

# This function plots the training and testing loss and MAE histories over epochs.
def plot_training_history(TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE):
    
    # Input Arguments:
    # TRAIN_LOSS : list storing the training loss values for each epoch.
    # TRAIN_MAE :  list storing the training MAE values for each epoch.
    # TEST_LOSS :  list storing the testing loss values for each epoch.
    # TEST_MAE :   list storing the testing MAE values for each epoch.
    
    # Use dark background.
    plt.style.use('dark_background')
    
    # 1) Plot Loss History
    plt.figure(figsize=(8,6))
    plt.plot(TRAIN_LOSS, label="Train Loss")
    plt.plot(TEST_LOSS, label="Test Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    # 2) Plot MAE History
    plt.figure(figsize=(8,6))
    plt.plot(TRAIN_MAE, label="Train MAE")
    plt.plot(TEST_MAE, label="Test MAE")
    plt.title("MAE per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# This function plots actual and predicted values for both training and testing
# partitions on a single figure.
def plot_predictions(train_preds, train_actuals, test_preds,  test_actuals):
    
    # Input Arguments:
    # train_preds : numpy array storing model predictions on the training set 
    #               (scaled or unscaled).
    # train_actuals : numpy array storing ground-truth (actual) target values 
    #                 for the training set (scaled or unscaled).
    # test_preds : numpy array storing model predictions on the test set 
    #              (scaled or unscaled).
    # test_actuals : numpy array storing ground-truth (actual) target values 
    #                for the test set (scaled or unscaled).
    

    # Convert all inputs to 1D NumPy arrays
    train_preds = np.array(train_preds).reshape(-1)
    train_actuals = np.array(train_actuals).reshape(-1)
    test_preds = np.array(test_preds).reshape(-1)
    test_actuals = np.array(test_actuals).reshape(-1)
    
    # Determine the indices for training and testing
    n_train = len(train_preds)
    n_test  = len(test_preds)
    
    # Training set on x-axis: [0, 1, 2, ..., n_train-1]
    train_x = np.arange(n_train)
    
    # Testing set on x-axis: [n_train, ..., n_train + n_test - 1]
    test_x = np.arange(n_train, n_train + n_test)

    # Use dark backgroud.
    plt.style.use('dark_background')

    plt.figure(figsize=(8,6))

    # Plot training actual and predicted
    plt.plot(train_x, train_actuals, label="Train Actual", alpha=0.7)
    plt.plot(train_x, train_preds,   label="Train Pred",   alpha=0.7)

    # Plot testing actual and predicted
    plt.plot(test_x, test_actuals, label="Test Actual", alpha=0.7)
    plt.plot(test_x, test_preds,   label="Test Pred",   alpha=0.7)

    plt.title("Model Predictions vs Actual Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
   
# This function reports the names, shapes, and total count of learnable 
# parameters in a PyTorch model.    
def report_model_parameters(model):
    
    # Input Arguments:
    # -----------
    # model : torch.nn.Module
    #         The PyTorch neural network model whose parameters will be inspected.

    # Output Arguments:
    # --------
    # None
    print("Model Parameters:")
    
    total_params = 0  # Counter for the total number of learnable parameters

    # Loop over all named parameters of the model
    for name, param in model.named_parameters():
        shape = tuple(param.shape)  # Get the shape as a tuple
        count = param.numel()       # Get the number of elements in the tensor
        total_params += count       # Accumulate total parameter count
        print(f"{name:40s} -> {shape}")

    print(f"\nTotal number of learnable parameters: {total_params}")

# This function visualizes 2D weight matrices (not biases) from a PyTorch 
# Feedforward or LSTM model. 
def visualize_model_weights(model, max_dim = 256, annotate = False):
    
    # Input Arguments:
    # -----------
    # model : torch.nn.Module
    #         A trained PyTorch model (e.g., feedforward or LSTM).
    #
    # max_dim : int
    #    Maximum allowed size of weight matrices in either dimension for visualization.
    #
    # annotate : bool
    #    If True, display numerical values in the heatmap (only useful for small matrices).

    # Output Arguments:
    # --------
    # None
    #    The function displays Seaborn heatmaps of weight matrices for inspection.
    
    # Set up dark background style
    plt.style.use('dark_background')
    sns.set_theme(style="dark", rc={
        "axes.facecolor": "#111111",
        "figure.facecolor": "#111111",
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#dddddd",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "text.color": "#ffffff",
        "axes.titlecolor": "#ffffff"
    })

    # Extract 2D weight matrices only
    weights = [(name, param) for name, param in model.named_parameters()
               if "weight" in name and param.dim() == 2]

    if not weights:
        print("No 2D weight matrices found in model.")
        return

    # Layout for subplots
    num = len(weights)
    cols = min(3, num)
    rows = math.ceil(num / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), facecolor="#111111")
    axes = axes.flatten() if num > 1 else [axes]

    for i, (name, param) in enumerate(weights):
        w = param.detach().cpu().numpy()

        if w.shape[0] > max_dim or w.shape[1] > max_dim:
            axes[i].set_title(f"{name} (skipped)")
            axes[i].axis('off')
            continue

        # Z-score normalization: (W - mean) / std
        mean, std = np.mean(w), np.std(w)
        if std > 0:
            w_norm = (w - mean) / std
        else:
            w_norm = np.zeros_like(w)

        # Clip z-scores to [-2, 2] to avoid outlier dominance
        w_clipped = np.clip(w_norm, -2, 2)

        # Normalize to [0, 1] after clipping for consistent colormap mapping
        w_final = (w_clipped + 2) / 4  # [-2,2] → [0,1]

        # Use perceptually uniform colormap suited for dark backgrounds
        sns.heatmap(w_final, ax=axes[i], cmap="viridis", cbar=True,
                    vmin=0, vmax=1, annot=annotate, fmt=".2f",
                    linewidths=0.05, linecolor='#222222')

        axes[i].set_title(name)
        axes[i].set_xlabel("Input Features")
        axes[i].set_ylabel("Output Units")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
