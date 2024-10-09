### This program takes in the level 1 radiometer data
### and processes it to level 2 using the hotfix neural networks.
### This is to handle the band instability and does NOT
### include calibration for the Valparaiso area.
###
### Christopher Phillips
### Valparaiso univ.
### Geography and Meteorology Dept.
### Oct. 2024

##### START OPTIONS #####

# Paths to the model weights
Tmodel_weights = 'models/temp_network_20241005_05.pt'
RHmodel_weights = 'models/rh_network_20241005_v01.pt'
Vmodel_weights = 'models/vapor_network_20241005_01.pt'
Lmodel_weights = 'models/liquid_network_20241005_01.pt'

# Directory with level 1 data
rdir = '/archive/campus_mesonet_data/rooftop_radiometer'

# Directory to which to save processed data
sdir = '/archive/campus_mesonet_data/rooftop_radiometer/processed_data'

#####  END OPTIONS  #####

### Import required modules
from src import readers
from src.tools import filter_level_1 as filter
from src.neuralnet import temp_net
from src.neuralnet import rh_net

from datetime import datetime
from glob import glob
import numpy as np
import netCDF4 as nc
import torch

### Load the models
Tmodel = temp_net()
Tmodel.load_state_dict(torch.load(Tmodel_weights, weights_only=True))
Tmodel.eval()
RHmodel = temp_net()
RHmodel.load_state_dict(torch.load(RHmodel_weights, weights_only=True))
RHmodel.eval()
Lmodel = temp_net()
Lmodel.load_state_dict(torch.load(Lmodel_weights, weights_only=True))
Lmodel.eval()
Vmodel = temp_net()
Vmodel.load_state_dict(torch.load(Vmodel_weights, weights_only=True))
Vmodel.eval()

### Process the Level 1 data into inputs for the model
# Find the file and read it in
date = datetime.utcnow()
print(f'{rdir}/{date.strftime("%Y-%m-%d")}_*_lv1.csv')
f1 = glob(f'{rdir}/{date.strftime("%Y-%m-%d")}_*_lv1.csv')[0]
print(f1)
exit()
data1 = readers.read_level_1(f1)

# Filter it
filtered_data1, qc1 = filter(data1, sigma=2.0)
qc1 = ~np.any(qc1, axis=0)
print(filtered_data1.shape)
exit()

### Run the models

### Write the new Level 2 data