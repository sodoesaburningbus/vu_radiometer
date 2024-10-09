### This program trains the neural network
### for the Valparaiso radiometer hotfix.
### Just select which network to train (temperature, humidity, liquid, or vapor)
### and run the program.
###
### Christopher Phillips
### Geography and Meteorology
### Valparaiso Univ.
### Oct. 2024

##### START OPTIONS #####

# Number of epochs
n_epochs = 1000

# Batch size
batch_size = 100

# Fraction of data to reserve for validation [0,1]
split = 0.15

# Model to train (temp, rh, liquid, vapor)
model_type = 'rh'

# Location of the training data
input_dir = '../training_data/hotfix'

# Location to save the model file
spath = f'models/{model_type}_network_20241005_v01.pt'

# Location to save the log file
log_path = f'models/{model_type}_network_training_log_20241005_v01.txt'

# Optional seed for reproducibility (None if none)
seed = 44

#####  END OPTIONS  #####

### Import modules
import copy
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Import the model
if (model_type == 'temp'):
    from src.neuralnet import temp_net
    model = temp_net()
else:
    from src.neuralnet import rh_net
    model = rh_net()

# Load in the training data
X = np.load(f'{input_dir}/inputs.npy')
y = np.load(f'{input_dir}/{model_type}_targets.npy')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(1.0-split), shuffle=True, random_state=seed)

print('Training data shape')
print('X_train', X_train.shape)
print('y_train', y_train.shape)

# Normalize
Xbar = np.mean(X_train)
Xsigma = np.std(X_train)
ybar = np.mean(y_train)
ysigma = np.std(y_train)

X_train = (X_train-Xsigma)/Xbar
X_test = (X_test-Xsigma)/Xbar
y_train = (y_train-ysigma)/ybar
y_test = (y_test-ysigma)/ybar

# Convert to PyTorch Tensors
# Note the 32bit dtype. Model weights default to float32 and the data must be the same
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Get training start date
start_date = datetime.utcnow()

# Create the log file
log = open(log_path, 'w')
log.write(f'Number of epochs: {n_epochs}\nBatch size: {batch_size}\nStart Date: {start_date.strftime("%Y-%m-%d_%H%M")}\nEpoch, Test Loss, Training Loss')

# Set the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Perform the actual training
best_loss = np.inf # Initialize a best loss metric
for epoch in range(n_epochs): # The Epoch loop. Each Epoch will loop through all training data
    
    # Set the model to training mode (some layers behave differntly during training)
    model.train()

    # The batch loop, break the data into chunks for training
    for start in torch.arange(0, len(X_train), batch_size):

        # Get training batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]

        # Do a forward pass through the model
        # This is where the model makes prediction
        y_pred = model(X_batch)

        # Compute the loss
        batch_loss = loss_fn(y_pred, y_batch)

        # Back propgation
        # This is where the model evaluates it performance
        optimizer.zero_grad()
        batch_loss.backward()

        # Update the model weights
        optimizer.step()

    # Evaluate model accuracy after the Epoch
    model.eval() # Set model to evaluation mode
    y_pred = model(X_test)

    loss = loss_fn(y_pred, y_test)
    loss = float(loss)
    log.write(f'\n{epoch},{loss:.7f},{batch_loss:.7f}')

    # Check if best accuracy
    if (loss < best_loss):
        best_loss = loss
        best_weights = copy.deepcopy(model.state_dict())

# Get the training end date
end_date = datetime.utcnow()
log.write(f'\n\nTotal training time: {(end_date-start_date).total_seconds()/60.0:.2f} minutes')

# After training, restore model to best performance and save
model.load_state_dict(best_weights)
torch.save(model.state_dict(), spath)

# Unscale validation data to compute final error statistics
y_pred = model(X_test).detach().numpy()
y_test = y_test.detach().numpy()
y_test = (y_test*ybar)+ysigma
y_pred = (y_pred*ybar)+ysigma

# Total RMSE
rmse = np.sqrt(np.mean((y_pred-y_test)**2))
mbe = np.mean(y_pred-y_test)
mae = np.mean(np.abs(y_pred-y_test))

log.write(f'\n\nFinal Stats\nRMSE = {rmse:.3f}\nMBE = {mbe:.3f}\nMAE = {mae:.3f}\nPE = {rmse/ybar*100.0:.3f} %')
log.close()