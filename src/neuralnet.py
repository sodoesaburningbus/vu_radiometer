### This class file contains the neural networks for the Valparaiso radiometer
### They convert the observed brightness temperatures into the different atmospheric variables.
###
### Christopher Phillips
### Valparaiso university
### Dept. of Geography and Meteorology
### github.com/sodoesaburningbus

### Import required modules
import torch
import torch.nn as nn
import torch.nn.functional as F

### The temperature network
class temp_net(nn.Module):

    # Initialize
    def __init__(self):

        # Setup pytorch class framework
        super(temp_net, self).__init__()

        # Define the layers
        self.layer1 = nn.Linear(45, 90)
        self.layer2 = nn.Linear(90, 120)
        self.layer3 = nn.Linear(120, 160)
        self.layer4 = nn.Linear(160, 200)
        self.layer5 = nn.Linear(200, 160)
        self.layer6 = nn.Linear(160, 120)
        self.layer7 = nn.Linear(120, 90)
        self.layer8 = nn.Linear(90, 58)

        return

    # The forward function
    def forward(self, X):

        # Apply the fully-connected layers with leaky-ReLU
        X = F.leaky_relu(self.layer1(X))
        X = F.leaky_relu(self.layer2(X))
        X = F.leaky_relu(self.layer3(X))
        X = F.leaky_relu(self.layer4(X))
        X = F.leaky_relu(self.layer5(X))
        X = F.leaky_relu(self.layer6(X))
        X = F.leaky_relu(self.layer7(X))

        # The output layer
        X = self.layer8(X)

        return X

### The relative humidity network
class rh_net(nn.Module):

    # Initialize
    def __init__(self):

        # Setup pytorch class framework
        super(rh_net, self).__init__()

        # Define the layers
        self.layer1 = nn.Linear(45, 90)
        self.layer2 = nn.Linear(90, 120)
        self.layer3 = nn.Linear(120, 160)
        self.layer4 = nn.Linear(160, 200)
        self.layer5 = nn.Linear(200, 160)
        self.layer6 = nn.Linear(160, 120)
        self.layer7 = nn.Linear(120, 90)
        self.layer8 = nn.Linear(90, 58)

        return

    # The forward function
    def forward(self, X):

        # Apply the fully-connected layers with leaky-ReLU
        X = F.leaky_relu(self.layer1(X))
        X = F.leaky_relu(self.layer2(X))
        X = F.leaky_relu(self.layer3(X))
        X = F.leaky_relu(self.layer4(X))
        X = F.leaky_relu(self.layer5(X))
        X = F.leaky_relu(self.layer6(X))
        X = F.leaky_relu(self.layer7(X))

        # The output layer
        X = self.layer8(X)

        return X