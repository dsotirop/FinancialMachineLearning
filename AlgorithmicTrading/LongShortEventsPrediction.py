# Import required Python libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# =============================================================================
#                    Functions Definition Section
# =============================================================================

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

# =============================================================================
# Function to load stock data from CSV file
# =============================================================================
def load_stock_data(filename, directory):
    # Construct full file path
    file_path = os.path.join(directory, filename)
    
    # Read CSV with two-level header, use first column as datetime index
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    
    # Collapse multi-level columns to single level (e.g., keep 'Close' and 'Volume')
    df.columns = df.columns.get_level_values(0)
    
    # Extract Close and Volume columns only
    df_filtered = df[['Close', 'Volume']].copy()
    
    # Reset index to include Date as a column
    df_filtered = df_filtered.reset_index().rename(columns={'index': 'Date'})
    
    # Sort data by Date to ensure chronological order
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
    return df_filtered

# =============================================================================
# Function to compute future relative volume long/short signals
# =============================================================================
def compute_relative_volume_signals(df, past_time_window, future_time_window, alpha):
    results = []
    # Define the valid range for evaluation (we skip initial and final rows)
    valid_range_start = past_time_window
    valid_range_end = len(df) - future_time_window
    
    for i in range(valid_range_start, valid_range_end):
        # Current reference price
        current_price = df.loc[i, 'Close']
        
        # Look-ahead window of future prices and volumes
        future_prices = df.loc[i+1:i+future_time_window, 'Close']
        future_volumes = df.loc[i+1:i+future_time_window, 'Volume']

        # Define long and short conditions based on alpha threshold
        long_mask = future_prices > current_price * (1 + alpha)
        short_mask = future_prices < current_price * (1 - alpha)

        # Total volume in future window
        vol_total = future_volumes.sum()
        # Volume where price rises above alpha threshold
        vol_long = future_volumes[long_mask].sum()
        # Volume where price drops below -alpha threshold
        vol_short = future_volumes[short_mask].sum()

        # Compute relative volume ratios (avoid division by zero)
        rel_vol_long = vol_long / vol_total if vol_total > 0 else 0.0
        rel_vol_short = vol_short / vol_total if vol_total > 0 else 0.0

        # Append index and computed values
        results.append([i, rel_vol_long, rel_vol_short])
    
    # Return as DataFrame
    return pd.DataFrame(results, columns=['Index', 'RelVol_Long', 'RelVol_Short'])

# =============================================================================
# Function to prepare lagged input-output pairs
# =============================================================================
def prepare_lagged_data(df, signal_df, past_time_window):
    X_all, Y_all = [], []
    for _, row in signal_df.iterrows():
        idx = int(row['Index'])
        window = df[idx - past_time_window:idx, :]
        X_all.append(window)
        Y_all.append([row['RelVol_Long'], row['RelVol_Short']])
    return np.array(X_all), np.array(Y_all)

# =============================================================================
# Custom PyTorch Dataset (assumes precomputed X, Y)
# =============================================================================
class StockLSTMWindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =============================================================================
# Hybrid Conv1D + Attention Model for RelVol Signals
# =============================================================================

# This module implements a single-head self-attention mechanism over a sequence.
# It learns how much attention each time step in the input sequence should pay to others.
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Linear projections to obtain queries, keys, and values from input embeddings
        self.q = nn.Linear(embed_dim, embed_dim)  # Projects input to query vectors
        self.k = nn.Linear(embed_dim, embed_dim)  # Projects input to key vectors
        self.v = nn.Linear(embed_dim, embed_dim)  # Projects input to value vectors
        self.scale = embed_dim ** 0.5             # Scaling factor to normalize dot product

    def forward(self, x):
        # x has shape (batch_size, seq_len, embed_dim)
        Q = self.q(x)  # (B, L, D)
        K = self.k(x)  # (B, L, D)
        V = self.v(x)  # (B, L, D)

        # Compute dot-product attention scores between each time step pair
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, L, L)

        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)  # (B, L, L)

        # Use attention weights to weight the value vectors
        output = torch.bmm(weights, V)  # (B, L, D)
        return output, weights  # Return both output and attention weights (for inspection)

# This model combines 1D convolutional heads for 'Close' and 'Volume' time series
# with a self-attention layer, followed by global pooling and an MLP for final prediction.
class ConvAttentionRelVolPredictor(nn.Module):
    def __init__(self, input_channels=1, seq_len=100, embed_dim=64):
        super().__init__()

        # Convolutional head for 'Close' prices:
        self.conv_price = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),  # First conv layer
            nn.ReLU(),                                                # Activation
            nn.Conv1d(16, 32, kernel_size=3, padding=1),              # Second conv layer
            nn.ReLU()                                                 # Activation
        )

        # Convolutional head for 'Volume':
        self.conv_volume = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Projection layer to unify the concatenated feature maps into a common embedding space
        self.project = nn.Linear(64, embed_dim)  # 32 (Close) + 32 (Volume) = 64

        # Self-attention module operating on the sequence of embeddings
        self.attn = SelfAttention(embed_dim)

        # Global average pooling layer to reduce the sequence dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Reduces (B, D, L) → (B, D)

        # Final multilayer perceptron for regression
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: [RelVol_Long, RelVol_Short]
        )

    def forward(self, x):
        # x has shape (B, L, 2): 2 features → Close and Volume
        x_price = x[:, :, 0].unsqueeze(1)   # (B, 1, L): isolate Close
        x_volume = x[:, :, 1].unsqueeze(1)  # (B, 1, L): isolate Volume

        # Apply convolution separately on each signal
        f_price = self.conv_price(x_price)     # (B, 32, L)
        f_volume = self.conv_volume(x_volume)  # (B, 32, L)

        # Concatenate along channel dimension and reshape for attention
        x_feat = torch.cat([f_price, f_volume], dim=1).permute(0, 2, 1)  # (B, L, 64)

        # Project to embedding dimension for attention
        x_embed = self.project(x_feat)  # (B, L, embed_dim)

        # Apply self-attention over the time dimension
        attn_out, _ = self.attn(x_embed)  # (B, L, embed_dim)

        # Permute for pooling: (B, embed_dim, L)
        attn_out = attn_out.permute(0, 2, 1)

        # Apply global average pooling across time steps
        pooled = self.global_pool(attn_out).squeeze(-1)  # (B, embed_dim)

        # Final prediction using MLP
        return self.mlp(pooled)  # (B, 2)

# =============================================================================
# Training Loop for Attention-Based Regression
# =============================================================================
def train_model(model, train_loader, test_loader, device, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                preds = model(X_batch)
                loss = loss_fn(preds, Y_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")


# =============================================================================
# Prediction and Plotting Function
# =============================================================================
def evaluate_and_plot(model, loader, device, title_prefix):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_pred.append(preds)
            y_true.append(Y_batch.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    r2_long = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_short = r2_score(y_true[:, 1], y_pred[:, 1])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(y_true))

    axes[0].plot(t, y_true[:, 0], label='True RelVol_Long', linewidth=1.5)
    axes[0].plot(t, y_pred[:, 0], label=f'Predicted RelVol_Long (R²={r2_long:.2f})', 
                 linewidth=1.5, linestyle='--')
    axes[0].set_title(f'{title_prefix} RelVol_Long')
    axes[0].set_ylabel('Relative Volume')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(t, y_true[:, 1], label='True RelVol_Short', linewidth=1.5)
    axes[1].plot(t, y_pred[:, 1], label=f'Predicted RelVol_Short (R²={r2_short:.2f})', 
                 linewidth=1.5, linestyle='--')
    axes[1].set_title(f'{title_prefix} RelVol_Short')
    axes[1].set_ylabel('Relative Volume')
    axes[1].set_xlabel('Sample Index')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return r2_long, r2_short

# =============================================================================
#                    Main Code Section
# =============================================================================

# =============================================================================
# Step 0: Set the execution device.
device = get_execution_device()
# =============================================================================

# =============================================================================
# Step 1: Set data directory and data file to be loader later.
data_dir = "./data"
data_file = "TSLA_data.csv"
# =============================================================================

# =============================================================================
# Step 2: Load the designated data file.
stock_df = load_stock_data(data_file, data_dir)
# =============================================================================

# =============================================================================
# Step 3: Set the fundamental time-window related parameters of the model along
#         with the alpha parameter that determines the threshold for long and 
#         short events.
#         - Compare future prices to current price:
#         - Long Condition: P_future > P_now * (1 + alpha)
#         - Short Condition: P_future < P_now * (1 - alpha)
#         - Neutral Condition: All others
past_time_window = 100
future_time_window = 50
alpha = 0.02
# =============================================================================

# =============================================================================
# Step 4: Compute the relative volume signals that will act as the target 
#         regression variables.
signal_df = compute_relative_volume_signals(stock_df, past_time_window, 
                                            future_time_window, alpha)
# =============================================================================

# =============================================================================
# Step 5: Prepare raw data arrays.
raw_inputs = stock_df[['Close', 'Volume']].values
# =============================================================================

# =============================================================================
# Step 6: Prepare lagged input-output pairs.
X_all, Y_all = prepare_lagged_data(raw_inputs, signal_df, past_time_window)
# =============================================================================

# =============================================================================
# Step 7: Normalize based on training inputs only by settin first the test size.
test_size = 0.2
train_idx, test_idx = train_test_split(np.arange(len(X_all)), 
                                       test_size=test_size, 
                                       random_state=42)
scaler = StandardScaler().fit(X_all[train_idx].reshape(-1, 2))
X_train = scaler.transform(X_all[train_idx].reshape(-1, 2)).reshape(-1, 
                                                                    past_time_window, 
                                                                    2)
X_test = scaler.transform(X_all[test_idx].reshape(-1, 2)).reshape(-1, 
                                                                  past_time_window, 
                                                                  2)
Y_train = Y_all[train_idx]
Y_test = Y_all[test_idx]
# =============================================================================

# =============================================================================
# Step 8: Set the training-related parameters.
batch_size = 32
epochs = 100
embed_dim = 64
# =============================================================================

# =============================================================================
# Step 8: Wrap datasets.
train_dataset = StockLSTMWindowDataset(X_train, Y_train)
test_dataset = StockLSTMWindowDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# =============================================================================

# =============================================================================
# Step 9: Initialize model.
# =============================================================================
model = ConvAttentionRelVolPredictor(input_channels=1, seq_len=past_time_window, 
                                     embed_dim=embed_dim).to(device)

# =============================================================================
# Step 10: Model training and testing.
# =============================================================================
train_model(model, train_loader, test_loader, device, epochs=epochs)
# =============================================================================

# =============================================================================
# Step 11: Evaluate and plot predictions.
r2_train = evaluate_and_plot(model, train_loader, device, "Training")
r2_test = evaluate_and_plot(model, test_loader, device, "Testing")
# =============================================================================