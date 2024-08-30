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


### The temeprature network
class temp_net(nn.Module):

    # Initialize
    def __init__(self):

        # Setup pytorch class framework
        super(temp_net, self).__init__()

        # Define the layers
        self.layer1 = nn.Linear(45, 180)
        self.layer2 = nn.Linear(180, 120)
        self.layer3 = nn.Linear(120 58)
        
        return