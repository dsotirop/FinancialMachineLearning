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
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import numpy as np
from auxiliary.utils import get_execution_device, plot_predictions
from auxiliary.utils import plot_training_history, report_model_parameters
from auxiliary.utils import visualize_model_weights
# ============================================================================= 
#                   CLASSES DEFINITION SECTION:
# =============================================================================

# This class provides fundamental functionality for the implementation of a 
# simple Dataset for time series data that does not allow random shuffling.
# Each item is a single (features, target) pair at a given time index.
class TimeSeriesDataset(Dataset):

    def __init__(self, X, y, device):
        
        # Input Arguments:
        #    X (np.ndarray or Tensor): shape (N, num_features)
        #    y (np.ndarray or Tensor): shape (N,) or (N, 1)
        #    device (torch.device): the device on which tensors will be placed.
        
        # Convert to tensors and move to device
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        # Ensure y is 2D for regression: (N, 1)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.X = X.to(device)
        self.y = y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# This class provides the implementation of a simple multi-layer perceptron.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_prob=0.2):
        

        # Input Arguments:
        #    input_dim (int): Size of the input features.
        #    hidden_layers (list of int): List containing the size of each hidden layer,
        #                                 e.g. [64, 32].
        #    dropout_prob (float): Probability of dropping out neurons after
        #                          each hidden layer. Default: 0.2
        super(MLP, self).__init__()
        layers = []
        # Start with the input dimension
        in_features = input_dim

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.Sigmoid()) # Or nn.ReLU()
            layers.append(nn.Dropout(p=dropout_prob))  # <-- Dropout added here
            in_features = hidden_size

        # Output layer (1 neuron for regression)
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# This class provides the implementation of a simple linear neural network where
# its output is formed as a weighted average of the input features plus the bias
# term: F(x) = W*x + b
class LinearNN(nn.Module):

    def __init__(self, input_dim):
        
        # Input Arguments:
        # input_dim: Integer indicating the size of the input features.
        
        super(LinearNN, self).__init__()
        # A single linear layer from input_dim to 1
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        
        # This function implements the forward pass through the linear layer.

        # Input Arguments:
        # x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        # Output Arguments:
        # torch.Tensor: Output tensor of shape (batch_size, 1).

        return self.linear(x)

# ============================================================================= 
#                   TRAINING / TESING FUNCTIONS DEFINITION SECTION:
# =============================================================================

# Define the function that performs the training of the neural network model.
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):

    # Input Arguments:
    # - model (nn.Module): The instance of the neural network model to be trained.
    # - train_loader (DataLoader): The DataLoader object that provides the batches of
    #   training feature vectors along with the corresponding target values.
    # - test_loader (DataLoader): The DataLoader object that provides the batches of
    #   testing feature vectors along with the corresponding target values.
    # - criterion (callable): The optimization criterion (e.g., MSELoss) to be
    #   minimized during training.
    # - optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam) used during
    #   training to update model parameters.
    # - epochs (int): The number of training epochs to be performed.

    # Output Arguments:
    #  * Average scaled train loss (e.g., MSE)
    #  * Average scaled train MAE
    #  * Average scaled test loss (e.g., MSE)
    #  * Average scaled test MAE
    
    # Initialize list containers for storing the training and testing loss and
    # mean absolute error (MAE).
    TRAIN_LOSS = []
    TRAIN_MAE = []
    TEST_LOSS = []
    TEST_MAE = []
    
    # Indicate that the following operations pertain to the training process.
    for epoch in range(epochs):
        # Set the model in training mode.
        model.train()

        # Initialize accumulators for total loss and total MAE over each epoch.
        total_loss = 0.0
        total_mae = 0.0

        # Loop through the various training batches stored in the train_loader.
        for X_batch, Y_batch in train_loader:
            # Set to zero the gradient vector w.r.t. the model's parameters.
            optimizer.zero_grad()

            # Obtain the predictions of the model for the current training batch.
            preds = model(X_batch)

            # Compute the current loss for the model on this batch.
            loss = criterion(preds, Y_batch)

            # Perform the backward pass.
            loss.backward()

            # Update model parameters.
            optimizer.step()

            # Accumulate the loss.
            total_loss += loss.item()

            # Compute the MAE for this batch and accumulate it.
            batch_mae = torch.mean(torch.abs(preds - Y_batch)).item()
            total_mae += batch_mae

        # Compute the average loss and average MAE for the current training epoch.
        avg_train_loss = total_loss / len(train_loader)
        avg_train_mae = total_mae / len(train_loader)

        # Compute the average test loss and average test MAE after this epoch.
        avg_test_loss, avg_test_mae = test_model(model, test_loader, criterion)
        
        # Store the accuracy metrics for the current batch.
        TRAIN_LOSS.append(avg_train_loss)
        TRAIN_MAE.append(avg_train_mae)
        TEST_LOSS.append(avg_test_loss)
        TEST_MAE.append(avg_test_mae)

        # Report the training and testing metrics every epoch.
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss (scaled): {avg_train_loss:.6f}, "
            #f"Train MAE (scaled): {avg_train_mae:.6f}, "
            f"Test Loss (scaled): {avg_test_loss:.6f}, "
            #f"Test MAE (scaled): {avg_test_mae:.6f}"
            )
        
    return TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE


# Define the function that performs the testing of the neural network model.
def test_model(model, test_loader, criterion):

    # Input Arguments:
    # - model (nn.Module): The instance of the trained neural network model
    #   to be evaluated.
    # - test_loader (DataLoader): The DataLoader object that provides the batches
    #   of testing feature vectors along with the corresponding target values.
    # - criterion (callable): The optimization criterion (e.g., MSELoss) to be
    #   evaluated on the test set.

    # Output Arguments:
    # - avg_test_loss (float): The average scaled loss (e.g., MSE) computed over
    #   the entire test set.
    # - avg_test_mae (float): The average scaled Mean Absolute Error computed
    #   over the entire test set.
    

    # Indicate that the following operations pertain to the testing process.
    model.eval()

    # Initialize a list to collect the test loss values for each testing batch.
    test_losses = []
    # Initialize an accumulator for total MAE.
    total_mae = 0.0

    # Perform the actual testing without computing gradients.
    with torch.no_grad():
        # Loop through the various testing batches stored in the test_loader.
        for X_batch, Y_batch in test_loader:
            # Obtain the predictions for the current testing batch.
            preds = model(X_batch)

            # Compute the current test loss.
            loss = criterion(preds, Y_batch)

            # Accumulate the loss for computing the average later.
            test_losses.append(loss.item())

            # Compute and accumulate the MAE for this batch.
            batch_mae = torch.mean(torch.abs(preds - Y_batch)).item()
            total_mae += batch_mae

    # Compute the average test loss over all testing batches.
    avg_test_loss = np.mean(test_losses)
    # Compute the average test MAE over all testing batches.
    avg_test_mae = total_mae / len(test_loader)

    # Return the average test loss and MAE.
    return avg_test_loss, avg_test_mae
    
# Define the function that returns all model predictions and respective actual
# target values. 
def get_predictions(model, data_loader):
    
    # Input Arguments:
    # model : The trained (or partially trained) PyTorch model.
    # data_loader : A PyTorch DataLoader containing your dataset split, e.g. 
    #               train or test.

    # Output Arguments:
    # all_preds : numpy.ndarray of shape (N, 1) storing model predictins for 
    #             all samples in the loader.
    # all_targets : numpy.ndarray of shape (N, 1) storing the ground-truth
    #               (actual) targets for all samples in the loader.
    

    # Put the model into evaluation mode (disable dropout, etc.)
    model.eval()

    # Lists to collect predictions and targets from each batch
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            # Forward pass
            preds = model(X_batch)

            # Move everything to CPU, then convert to NumPy
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    # Concatenate all batch results into a single NumPy array
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

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
data_file = 'IBM_time_series_data.csv'

# Construct the full data path
data_path = os.path.join(data_directory,data_file)

# Load the DataFrame from the CSV file
df = pd.read_csv(data_path)

# Drop Date column
df = df.drop(columns=['Date'])

# Separate target regression variables from features. In the context of this
# implementation each data file stores the target regression variable as the 
# first series object in the respective dataframe.
target_col = df.columns[0]
X = df.drop(columns=[target_col]).values  # shape: (N, num_features)
Y = df[target_col].values                 # shape: (N,) 

# Perform training / testing splitting at this point to avoid data leakage 
# during scaling
train_ratio = 0.8
n_train = int(len(X) * train_ratio)
X_train_raw, X_test_raw = X[:n_train], X[n_train:]
Y_train_raw, Y_test_raw = Y[:n_train], Y[n_train:]

# Scale independent regression variables stored as X.
x_scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1))
X_train_scaled = x_scaler.fit_transform(X_train_raw)
X_test_scaled  = x_scaler.transform(X_test_raw)

# Scale dependebt regression variables stored as Y ensuring that the respective
# numpy arrays are 2-dimensional before scaling.
y_scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = y_scaler.fit_transform(Y_train_raw.reshape(-1, 1))
Y_test_scaled  = y_scaler.transform(Y_test_raw.reshape(-1, 1))

# ============================================================================= 
# Step 2: Create Datasets & DataLoaders
# =============================================================================

# Get the execution device to be used for training and testing the model.
device = get_execution_device()

# Set the batch size to be used during training and testing.
# batch_size = 32

train_dataset = TimeSeriesDataset(X_train_scaled, Y_train_scaled, device=device)
test_dataset  = TimeSeriesDataset(X_test_scaled, Y_test_scaled, device=device)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# ============================================================================= 
# Step 3: Model Definition
# =============================================================================

# Get the dimensionality of the input features.
input_dim = X_train_scaled.shape[1]

# Define the number of neurons per hidden layer of the network.
hidden_layers =  [128,64,32,16]

# Instantiate the feed forward neural network object.
model = MLP(input_dim, hidden_layers).to(device)

# Instantiate the linear network object.
# model = LinearNN(input_dim).to(device)

# ============================================================================= 
# Step 4: Loss and Optimizer Definition
# =============================================================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ============================================================================= 
# Step 5: Model Training and Testing
# =============================================================================

# Set the number of training epochs.
epochs =  1000

# Perform the actual training process.
TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE = train_model(model, 
                                                         train_loader, 
                                                         test_loader, 
                                                         criterion, 
                                                         optimizer, 
                                                         epochs)
# Plot the training and testing history.
plot_training_history(TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE)

# Get predictions for the training set.
train_preds, train_actuals = get_predictions(model, train_loader)

# Get predictions for the testing set.
test_preds, test_actuals = get_predictions(model, test_loader)

# Obtain the unscaled versions of the training and testing predictions and 
# corresponding ground truth values. If the respective numpy arrays are shaped 
# (N,) they must be reshaped to (N,1) before using inverse_transform.
train_preds_unscaled = y_scaler.inverse_transform(train_preds.reshape(-1, 1))
train_actuals_unscaled = y_scaler.inverse_transform(train_actuals.reshape(-1, 1))
test_preds_unscaled = y_scaler.inverse_transform(test_preds.reshape(-1, 1))
test_actuals_unscaled = y_scaler.inverse_transform(test_actuals.reshape(-1, 1))

# Plot the actual and predicted target regression values utilizing the unscaled
# versions of the data.
plot_predictions(train_preds_unscaled, train_actuals_unscaled, 
                 test_preds_unscaled, test_actuals_unscaled)

# ============================================================================= 
# Step 6: Report Model Parameters & Visualize Weigtht Parameters
# =============================================================================
report_model_parameters(model)
visualize_model_weights(model)

# We may visualize distinct layers as:
# visualize_model_weights(model.network[0])
# visualize_model_weights(model.network[3]) 
