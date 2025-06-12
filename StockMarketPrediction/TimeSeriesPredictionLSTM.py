# =============================================================================
# This script file provides fundamental computational functionality for the 
# implementation of Case Study I: Supervised Learning Models for Stock Market
# Prediction through the utilization of LSTM Networks.
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

# Import required Python libraries.
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auxiliary.utils import get_execution_device, plot_training_history, plot_predictions

# ============================================================================= 
#                   CLASSES DEFINITION SECTION:
# =============================================================================

# -----------------------------------------------------------------------------
#                   Custom Dataset Class:
# -----------------------------------------------------------------------------
# Custom PyTorch Dataset for stock price prediction using past window sequences.
# Assumes data has already been scaled externally if needed.
# -----------------------------------------------------------------------------
class StockDataset(Dataset):

    def __init__(self, data, past_window_size, device):
        
        # Initializes the dataset with past window size.

        # Input Arguments:
        # - data: numpy array of shape (N, 1), already scaled if necessary
        # - past_window_size: number of past time steps to use as input
        # - device: torch device on which tensors should be placed

        # Output Arguments:
        # - None (initializes internal data structure)
        
        self.past_window_size = past_window_size
        self.data = data.astype(np.float32)
        self.device = device

    def __len__(self):
        
        # Returns the number of usable input-target pairs.

        # Output Arguments:
        # - Integer count of samples
        
        return len(self.data) - self.past_window_size

    def __getitem__(self, idx):
        
        # Retrieves input-target pair at index `idx`.

        # Input Arguments:
        # - idx: integer index

        # Output Arguments:
        # - x: FloatTensor of shape (past_window_size, 1)
        # - y: FloatTensor of shape (1,)
        
        x = self.data[idx:idx + self.past_window_size]
        y = self.data[idx + self.past_window_size]
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
               torch.tensor(y, dtype=torch.float32, device=self.device)

# -----------------------------------------------------------------------------
#                           LSTM Model Class:
# -----------------------------------------------------------------------------
# LSTM model for time series forecasting using past observations.
# -----------------------------------------------------------------------------
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        
        # Initializes the LSTM model.

        # Input Arguments:
        # - input_size: number of input features (1 for univariate)
        # - hidden_size: number of hidden units in LSTM
        # - num_layers: number of stacked LSTM layers

        # Output Arguments:
        # - None
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # If the input data is of shape (seq_len, batch_size, features) 
        # batch_first=True is not required by the LSTM to provide its output in
        # the form (seq_len, batch_size, hidden_size). If instead,  the input 
        # data is of shape (batch_size, seq_len, features) then batch_first=True 
        # is required so that the LSTM output will be of the desired shape 
        # (batch_size, seq_len, hidden_size).
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        # Forward pass of the model.

        # Input Arguments:
        # - x: Tensor of shape (batch_size, sequence_length, input_size)

        # Output Arguments:
        # - out: Tensor of shape (batch_size, 1)
        
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take output at last time step
        out = self.fc(out)
        return out

# ============================================================================= 
#                   TRAINING / TESING FUNCTIONS DEFINITION SECTION:
# =============================================================================

# -----------------------------------------------------------------------------
#                         Training Function
# -----------------------------------------------------------------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    
    # Trains the model and evaluates after each epoch.

    # Arguments:
    # - model: LSTM model
    # - train_loader: DataLoader for training data
    # - test_loader: DataLoader for testing data
    # - criterion: loss function
    # - optimizer: optimizer
    # - num_epochs: number of training epochs
    
    # Initialize list containers for storing the training and testing loss and
    # mean absolute error (MAE).
    TRAIN_LOSS = []
    TRAIN_MAE = []
    TEST_LOSS = []
    TEST_MAE = []
    
    for epoch in range(num_epochs):
        # Set the environment to training mode.
        model.train()
        total_loss = 0
        total_mae = 0

        for x, y in train_loader:
            output = model(x)
            loss = criterion(output, y)
            mae = torch.mean(torch.abs(output - y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mae += mae.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mae = total_mae / len(train_loader)

        avg_test_loss, avg_test_mae = test_model(model, test_loader, criterion)
        
        # Store the accuracy metrics for the current batch.
        TRAIN_LOSS.append(avg_train_loss)
        TRAIN_MAE.append(avg_train_mae)
        TEST_LOSS.append(avg_test_loss)
        TEST_MAE.append(avg_test_mae)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f} | "
              f"Test Loss: {avg_test_loss:.6f}, Test MAE: {avg_test_mae:.6f}")
        
    return TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE


# -----------------------------------------------------------------------------
#                         Testing Function
# -----------------------------------------------------------------------------
def test_model(model, test_loader, criterion):
    # Evaluates the model on the test dataset.

    # Arguments:
    # - model: trained LSTM model
    # - test_loader: DataLoader for test data
    # - criterion: loss function

    # Returns:
    # - test_loss: average loss
    # - test_mae: average mean absolute error
    
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            loss = criterion(output, y)
            mae = torch.mean(torch.abs(output - y))
            total_loss += loss.item()
            total_mae += mae.item()
    return total_loss / len(test_loader), total_mae / len(test_loader)

# -----------------------------------------------------------------------------
#                 Prediction Function (In the loop operation mode)
# -----------------------------------------------------------------------------
# Define the function that returns all model predictions and respective actual
# target values. 
def get_predictions(model, data_loader, scaler):
    
    # Generates predictions for all samples in the provided DataLoader.

    # Arguments:
    # - model: Trained PyTorch model.
    # - data_loader: DataLoader containing input-target pairs (batch_size=1 expected).
    # - scaler: The fitted scaler used for inverse transforming the predictions and targets.

    # Returns:
    # - all_preds: Numpy array of model predictions (inverse scaled).
    # - all_targets: Numpy array of ground truth values (inverse scaled).
    
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            pred = model(x)
            all_preds.append(pred.item())
            all_targets.append(y.item())

    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)

    all_preds = scaler.inverse_transform(all_preds).flatten()
    all_targets = scaler.inverse_transform(all_targets).flatten()

    return all_preds, all_targets

# ============================================================================= 
#                   MAIN CODE SECTION:
# =============================================================================

# ============================================================================= 
# Step 1: Load and preprocess data
# =============================================================================

# Set the name of the data directory.
data_directory = './data'
# The files that can be used are either the XXXX_time_series_data.csv which 
# contain as features lagged versions of only the target regression variable
# or XXXX_multi_time_series_data.csv which contain as features lagged versions
# of all the available data variables.
data_file = 'MSFT_time_series_data.csv'

# Construct the full data path.
data_path = os.path.join(data_directory,data_file)

# Load the DataFrame from the CSV file.
df = pd.read_csv(data_path)

# Drop Date column
df = df.drop(columns=['Date'])

# Acquire all dataframe series objects that store the value of a financial 
# instrument at t = 0.
columns_t0 = [col for col in df.columns if col.endswith('_t-0')]
time_series_df = df[columns_t0]

# Partition original dataset into training and testing subsets.
# -----------------------------------------------------------------------------
#                          IMPORTANT NOTE!!!!
# -----------------------------------------------------------------------------
# During training and testing partioning of the original time series data, no
# shuffling must be performed so that past and future values must not get 
# mixed.
# ----------------------------------------------------------------------------- 
test_ratio = 0.2
train_time_series_df , test_time_series_df = train_test_split(time_series_df, 
                                                            test_size=test_ratio, 
                                                            shuffle=False)

# Scale training data and transform test data.
scaler = StandardScaler()
train_time_series_scaled = scaler.fit_transform(train_time_series_df.values)
test_time_series_scaled = scaler.transform(test_time_series_df.values)

# ============================================================================= 
# Step 2: Create Datasets & DataLoaders
# =============================================================================

# Get the execution device to be used for training and testing the model.
device = get_execution_device()

# Set the past window size.
past_window_size = 20

# Set the training and testing batch sizes keeping in mind that the testing 
# batch size must be equal to 1.
train_batch_size = 64
test_batch_size = 1

# Create training and testing datasets and dataloaders.
train_dataset = StockDataset(train_time_series_scaled, 
                             past_window_size=past_window_size,
                             device=device)
test_dataset = StockDataset(test_time_series_scaled, 
                            past_window_size=past_window_size,
                            device=device)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# ============================================================================= 
# Step 3: Model Definition
# =============================================================================

# Initialize the parameters that define the internal structure of the neural 
# network model.
input_size = 1
hidden_size = 50
num_layers = 2

# Initialize the neural network model.
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, 
                  num_layers=num_layers).to(device)

# ============================================================================= 
# Step 4: Loss and Optimizer Definition
# =============================================================================

# Initialize the parameters correspoding to optimization criterion that will be
# used for training and the learning to be utilized.
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================================= 
# Step 5: Model Training and Testing
# =============================================================================

# Set the number of training epochs.
epochs =  100

# Perform the actual training process.
TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE = train_model(model, 
                                                         train_loader, 
                                                         test_loader, 
                                                         criterion, 
                                                         optimizer, 
                                                         epochs)
# Plot the training and testing history.
plot_training_history(TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE)

# Redefine train_loader for prediction mode where one future datapoint is 
# to be predicted per time.
train_loader_pred = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader_pred = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Get training and testing predictions.
train_preds, train_targets = get_predictions(model, train_loader_pred, scaler)
test_preds, test_targets = get_predictions(model, test_loader_pred, scaler)

# Plot training and testing predictions against their actual values.
plot_predictions(train_preds, train_targets, test_preds, test_targets)