# Import required Python libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# =============================================================================
#                    Functions Definition Section
# =============================================================================

# Define a function to get the correct training environemnt for the model.
def get_execution_device():
    if hasattr(torch.backends,"mps"):
        is_mps = torch.backends.mps.is_available()
    else:
        is_mps = False
    is_cuda = torch.cuda.is_available()
    if is_mps:
        device = torch.device("mps")
        print("MPS GPU is available!")
        print(70*"=")
    elif is_cuda:
        device = torch.device("cuda")
        print("CUDA GPU is available!")
        print(70*"=")
    else:
        device = torch.device("cpu")
        print("GPU is not available, CPU will be used instead!")
        print(70*"=")
    return device

# Function to load stock data from CSV file
def load_stock_data(filename, directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    df_filtered = df[['Close', 'Volume']].copy()
    df_filtered = df_filtered.reset_index().rename(columns={'index': 'Date'})
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
    return df_filtered

# Function to compute future relative volume long/short signals
def compute_relative_volume_signals(df, past_time_window, future_time_window, alpha):
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
        results.append([i, rel_vol_long, rel_vol_short])
    return pd.DataFrame(results, columns=['Index', 'RelVol_Long', 'RelVol_Short'])

# Function to prepare lagged data as input-output pairs
def prepare_lagged_data(df, signal_df, past_time_window):
    X_all, Y_all = [], []
    for _, row in signal_df.iterrows():
        idx = int(row['Index'])
        window = df[['Close', 'Volume']].iloc[idx - past_time_window:idx].values
        X_all.append(window)
        Y_all.append([row['RelVol_Long'], row['RelVol_Short']])
    return np.array(X_all), np.array(Y_all)

# Custom Dataset using preprocessed tensors
class StockLSTMWindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Self-Attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(weights, V)
        return output, weights

# Conv-Attention-MLP hybrid predictor
class ConvAttentionRelVolPredictor(nn.Module):
    def __init__(self,
                 input_channels=1,
                 conv_channels=[16, 32],
                 kernel_size=3,
                 embed_dim=64,
                 mlp_hidden_layers=[64],
                 output_dim=2):
        super().__init__()
        self.conv_price = self._make_conv_head(input_channels, conv_channels, kernel_size)
        self.conv_volume = self._make_conv_head(input_channels, conv_channels, kernel_size)
        final_channels = conv_channels[-1] * 2
        self.project = nn.Linear(final_channels, embed_dim)
        self.attn = SelfAttention(embed_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = self._make_mlp_head(embed_dim, mlp_hidden_layers, output_dim)

    def _make_conv_head(self, in_channels, channel_list, kernel_size):
        layers = []
        for out_channels in channel_list:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_mlp_head(self, input_dim, hidden_layers, output_dim):
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_price = x[:, :, 0].unsqueeze(1)
        x_volume = x[:, :, 1].unsqueeze(1)
        f_price = self.conv_price(x_price)
        f_volume = self.conv_volume(x_volume)
        x_feat = torch.cat([f_price, f_volume], dim=1).permute(0, 2, 1)
        x_embed = self.project(x_feat)
        attn_out, _ = self.attn(x_embed)
        attn_out = attn_out.permute(0, 2, 1)
        pooled = self.global_pool(attn_out).squeeze(-1)
        return self.mlp(pooled)

# Training function
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

# Evaluation and plotting with R^2

def evaluate_and_plot(model, loader, device, title_prefix=""):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(Y_batch.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    r2_long = r2_score(targets[:, 0], preds[:, 0])
    r2_short = r2_score(targets[:, 1], preds[:, 1])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(targets[:, 0], label='True RelVol_Long', linewidth=1.2)
    axes[0].plot(preds[:, 0], label=f'Predicted RelVol_Long (R²={r2_long:.2f})', linestyle='--')
    axes[0].set_title(f'{title_prefix} Relative Volume: Long Signal')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(targets[:, 1], label='True RelVol_Short', linewidth=1.2)
    axes[1].plot(preds[:, 1], label=f'Predicted RelVol_Short (R²={r2_short:.2f})', linestyle='--')
    axes[1].set_title(f'{title_prefix} Relative Volume: Short Signal')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return r2_long, r2_short

# =============================================================================
#                    Main Code Section
# =============================================================================

# Step 0: Set device
device = get_execution_device()

# Step 1: Load data
data_dir = "./data"
data_file = "TSLA_data.csv"
stock_df = load_stock_data(data_file, data_dir)

# Step 2: Compute relative volume targets
past_time_window = 100
future_time_window = 40
alpha = 0.02
signal_df = compute_relative_volume_signals(stock_df, past_time_window, future_time_window, alpha)

# Step 3: Prepare data
X_all, Y_all = prepare_lagged_data(stock_df, signal_df, past_time_window)
train_idx, test_idx = train_test_split(np.arange(len(X_all)), test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_all[train_idx].reshape(-1, 2))
X_train = scaler.transform(X_all[train_idx].reshape(-1, 2)).reshape(-1, past_time_window, 2)
X_test = scaler.transform(X_all[test_idx].reshape(-1, 2)).reshape(-1, past_time_window, 2)
Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]

# Step 4: Build datasets
train_dataset = StockLSTMWindowDataset(X_train, Y_train)
test_dataset = StockLSTMWindowDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Step 5: Train model
model = ConvAttentionRelVolPredictor(
    input_channels=1,
    conv_channels=[16, 32, 64],
    kernel_size=3,
    embed_dim=64,
    mlp_hidden_layers=[128, 64, 32],
    output_dim=2
).to(device)

train_model(model, train_loader, test_loader, device, epochs=200, lr=0.001)

# Step 6: Evaluate and visualize
r2_train = evaluate_and_plot(model, train_loader, device, title_prefix="Train")
r2_test = evaluate_and_plot(model, test_loader, device, title_prefix="Test")
