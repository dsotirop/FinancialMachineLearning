# ============================================================================
# LSTM-Based Adaptive Trading Agent with Indicator Signals
# ============================================================================
# This script implements an LSTM-based model for financial trading.
# It dynamically selects indicator window sizes and thresholds using
# Gumbel-Softmax and sigmoid functions, learns to generate trading
# signals, and optimizes for Sharpe ratio and cumulative returns.
#============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 1. Device Detection
# ============================================================================
def get_execution_device():
    if hasattr(torch.backends, "mps"):
        is_mps = torch.backends.mps.is_available()
    else:
        is_mps = False
    is_cuda = torch.cuda.is_available()
    if is_mps:
        device = torch.device("mps")
        print("MPS GPU is available!")
    elif is_cuda:
        device = torch.device("cuda")
        print("CUDA GPU is available!")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_execution_device()

# ============================================================================
# 2. LSTM Trading Agent with Debug Mode
# ============================================================================
class LSTMTradingAgent(nn.Module):
    def __init__(self, d_input, d_hidden, num_indicators, window_sizes, tau=1.0):
        super().__init__()
        self.num_indicators = num_indicators
        self.window_sizes = window_sizes
        self.M = len(window_sizes)
        self.tau = tau

        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True)

        self.window_heads = nn.ModuleList([
            nn.Linear(d_hidden, self.M) for _ in range(num_indicators)
        ])
        self.threshold_heads = nn.ModuleList([
            nn.Linear(d_hidden, 2) for _ in range(num_indicators)
        ])

        self.beta = nn.Parameter(torch.randn(num_indicators))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, indicator_bank, debug=False):
        B, T, d = x.shape
        if debug:
            print(f"[DEBUG] Input x shape: {x.shape}")  # [B, T, d_input]

        _, (h_n, _) = self.lstm(x)
        h_t = h_n[-1]  # [B, d_hidden]
        if debug:
            print(f"[DEBUG] LSTM final hidden state h_t shape: {h_t.shape}")

        signals = []
        for i in range(self.num_indicators):
            alpha_logits = self.window_heads[i](h_t)
            if debug:
                print(f"[DEBUG] Indicator {i}: alpha_logits shape: {alpha_logits.shape}")

            gumbel_noise = -torch.log(-torch.log(torch.rand_like(alpha_logits)))
            weights = F.softmax((alpha_logits + gumbel_noise) / self.tau, dim=-1)
            if debug:
                print(f"[DEBUG] Indicator {i}: Gumbel-Softmax weights shape: {weights.shape}")

            I_i = indicator_bank[i]
            if debug:
                print(f"[DEBUG] Indicator {i}: Indicator bank shape: {I_i.shape}")

            indicator_value = (weights * I_i).sum(dim=-1)
            if debug:
                print(f"[DEBUG] Indicator {i}: Weighted indicator value shape: {indicator_value.shape}")

            thresholds = self.threshold_heads[i](h_t)
            if debug:
                print(f"[DEBUG] Indicator {i}: thresholds shape: {thresholds.shape}")
            theta_plus, theta_minus = thresholds[:, 0], thresholds[:, 1]

            s_i = torch.sigmoid((indicator_value - theta_plus) / 0.1) - \
                  torch.sigmoid((theta_minus - indicator_value) / 0.1)
            if debug:
                print(f"[DEBUG] Indicator {i}: signal s_i shape: {s_i.shape}")
            signals.append(s_i)

        S = torch.stack(signals, dim=-1)
        if debug:
            print(f"[DEBUG] All signals stacked S shape: {S.shape}")

        decision = torch.tanh(S @ self.beta + self.bias)
        if debug:
            print(f"[DEBUG] Final decision shape: {decision.shape}")
        return decision

# ============================================================================
# 3. Custom Dataset
# ============================================================================
class TradingDataset(Dataset):
    def __init__(self, X, indicator_bank, returns):
        self.X = X.to(device)
        self.indicator_bank = [ib.to(device) for ib in indicator_bank]
        self.returns = returns.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        indicators = [bank[idx] for bank in self.indicator_bank]
        y = self.returns[idx]
        return x, indicators, y

# ============================================================================
# 4. Loss Functions
# ============================================================================
def sharpe_loss(returns):
    mean = returns.mean()
    std = returns.std(unbiased=False) + 1e-6
    return -mean / std

def return_loss(returns):
    return -returns.sum()

def hybrid_loss(returns, mu_sharpe=0.5, mu_ret=0.5):
    return mu_sharpe * sharpe_loss(returns) + mu_ret * return_loss(returns)

# ============================================================================
# 5. Training & Evaluation
# ============================================================================
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    all_returns = []

    for x, indicator_bank, y_true in dataloader:
        x = x.to(device)
        indicator_bank = [ind.to(device) for ind in indicator_bank]
        y_true = y_true.to(device)

        decision = model(x, indicator_bank, debug=False)
        predicted_returns = decision * y_true

        loss = hybrid_loss(predicted_returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_returns.append(predicted_returns.detach())

    returns = torch.cat(all_returns)
    return returns.mean().item(), returns.std().item(), -sharpe_loss(returns).item()

def evaluate(model, dataloader):
    model.eval()
    all_returns = []

    with torch.no_grad():
        for x, indicator_bank, y_true in dataloader:
            x = x.to(device)
            indicator_bank = [ind.to(device) for ind in indicator_bank]
            y_true = y_true.to(device)

            decision = model(x, indicator_bank, debug=False)
            predicted_returns = decision * y_true
            all_returns.append(predicted_returns)

    returns = torch.cat(all_returns)
    return returns.mean().item(), returns.std().item(), -sharpe_loss(returns).item()

# ============================================================================
# 6. Run Example
# ============================================================================
# Parameters
torch.manual_seed(42)
batch_size = 32
seq_len = 20
d_input = 6
d_hidden = 32
num_indicators = 4
window_sizes = [5, 10, 15, 20, 30]
epochs = 10

# Dummy data
N = 500
X = torch.randn(N, seq_len, d_input)
indicator_bank = [torch.randn(N, len(window_sizes)) for _ in range(num_indicators)]
true_returns = torch.randn(N)

# Dataset and Loader
dataset = TradingDataset(X, indicator_bank, true_returns)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Model and Optimizer
model = LSTMTradingAgent(d_input, d_hidden, num_indicators, window_sizes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Demonstration with debug ON (once only)
print("\n--- Debug Mode Output ---")
x_sample = X[:4].to(device)
indicator_sample = [ib[:4].to(device) for ib in indicator_bank]
_ = model(x_sample, indicator_sample, debug=True)

# Training Loop
print("\n--- Training ---")
for epoch in range(epochs):
    mean_ret, std_ret, sharpe = train_one_epoch(model, dataloader, optimizer)
    print(f"[Epoch {epoch+1}] Return: {mean_ret:.4f}, Std: {std_ret:.4f}, Sharpe: {sharpe:.4f}")

# Evaluate the model using the same dataloader (or a separate test loader)
mean_ret_eval, std_ret_eval, sharpe_eval = evaluate(model, dataloader)

print("\n--- Evaluation ---")
print(f"Mean Return: {mean_ret_eval:.4f}")
print(f"Return Std Dev: {std_ret_eval:.4f}")
print(f"Sharpe Ratio: {sharpe_eval:.4f}")