# This is a Python library containing the implementation for various auxiliary
# functions.

# Import required Python libraries.
import matplotlib.pyplot as plt
import torch
import numpy as np
# ============================================================================= 
#                   HELPER FUNCTIONS DEFINITION SECTION:
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
