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

# Working directory
wdir = '/archive/campus_mesonet_data/rooftop_radiometer/vu_radiometer'

# Paths to the model weights
Tmodel_weights = f'{wdir}/models/temp_network_v03.pt'
RHmodel_weights = f'{wdir}/models/rh_network_v03.pt'
Vmodel_weights = f'{wdir}/models/vapor_network_v03.pt'
Lmodel_weights = f'{wdir}/models/liquid_network_v03.pt'

# Directory with normalization files
ndir = f'{wdir}/models'

# Directory with level 1 data
rdir = '/archive/campus_mesonet_data/rooftop_radiometer'

# Directory to which to save processed data
sdir = '/archive/campus_mesonet_data/rooftop_radiometer/processed_data'

# Radiometer heights (do not change! unless you've re-trained the network to new ones.)
radiometer_heights = [
    0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.25, 2.5, 2.75, 3,
    3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
    5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8,
    8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10
]

#####  END OPTIONS  #####

### Import required modules
from src import readers
from src.tools import filter_level_1 as filter
from src.neuralnet import temp_net, rh_net

from datetime import datetime, timedelta
from glob import glob
import numpy as np
import netCDF4 as nc
import torch

from src import tools

### Load the models
Tmodel = temp_net()
Tmodel.load_state_dict(torch.load(Tmodel_weights, weights_only=True))
Tmodel.eval()
RHmodel = rh_net()
RHmodel.load_state_dict(torch.load(RHmodel_weights, weights_only=True))
RHmodel.eval()
Lmodel = rh_net()
Lmodel.load_state_dict(torch.load(Lmodel_weights, weights_only=True))
Lmodel.eval()
Vmodel = rh_net()
Vmodel.load_state_dict(torch.load(Vmodel_weights, weights_only=True))
Vmodel.eval()

### Process the Level 1 data into inputs for the model
# Find the file and read it in
date = datetime.utcnow()
f1 = glob(f'{rdir}/{date.strftime("%Y-%m-%d")}_*_lv1.csv')[0]
data1 = readers.read_level_1(f1)

# Filter it
filtered_data1, qc1 = filter(data1, sigma=2.0)
inputs = np.concatenate((filtered_data1[:,1:5],filtered_data1[:,9:]), axis=1)

### Create netCDF file for output
### Write the new Level 2 data
fn = nc.Dataset(f"{sdir}/radiometer_{f1.replace('lv1.csv', 'lv2.nc').split('/')[-1]}",'w')

# File attributes
fn.description = 'Thermodynamic profiles from Valparaiso Radiometer with channel instabilities corrected.'
fn.date_created = date.strftime("%Y-%m-%d %H:%M UTC")
fn.temp_net = Tmodel_weights.split('/')[-1]
fn.vapor_net = Vmodel_weights.split('/')[-1]
fn.humidity_net = RHmodel_weights.split('/')[-1]
fn.water_net = Lmodel_weights.split('/')[-1]

# Create the dimensions
time_dim = fn.createDimension('t', filtered_data1[:,0].size)  # 487 times
height_dim = fn.createDimension('z', len(radiometer_heights))  # 58 heights

# Create the variables
# Time
times = fn.createVariable('time', np.float32, ('t',))
times.long_name = 'Hours since midnight (UTC)'
times.units = 'Hours'
times[:] = filtered_data1[:,0]

# Height
heights = fn.createVariable('height', np.float32, ('z'))
heights.long_name = 'Height above radiometer'
heights.units = 'km'
heights[:] = np.array(radiometer_heights)

# PBLH
pblh_var = fn.createVariable('PBLH', np.float32, ('t'))
pblh_var.long_name = 'Boundary layer height'
pblh_var.units = 'km'

# Temperature
out_temps = fn.createVariable('TEMP', np.float32, ('t','z'))
out_temps.long_name = 'Temperature'
out_temps.units = 'K'
out_temps.missing_value = -999

# Relative Humidity
out_rh = fn.createVariable('RH', np.float32, ('t','z'))
out_rh.long_name = 'Relative Humidity'
out_rh.units = '%'
out_rh.missing_value = -999

# Water vapor density
out_vapor = fn.createVariable('VAPOR', np.float32, ('t','z'))
out_vapor.long_name = 'Water Vapor Density'
out_vapor.units = 'g/m3'
out_vapor.missing_value = -999

# Liquid water density
out_liquid = fn.createVariable('LIQUID', np.float32, ('t','z'))
out_liquid.long_name = 'Liquid Water Density'
out_liquid.units = 'g/m3'
out_liquid.missing_value = -999

### Run the models
# Loop over model types
for model, var in zip([Tmodel, RHmodel, Lmodel, Vmodel], ['temp', 'rh', 'liquid', 'vapor']):

    # Normalize the data
    norms = np.load(f'{ndir}/{var}_norms.npz')
    inputs = np.concatenate((filtered_data1[:,1:5],filtered_data1[:,9:]), axis=1)
    inputs = (inputs-norms['xbar'])/norms['xsigma']
    inputs = torch.tensor(inputs, dtype=torch.float32)

    output = model(inputs).detach().numpy()
    for i in range(norms['ybar'].size):
        output[:,i] = (output[:,i]*norms['ysigma'][i])+norms['ybar'][i]
    
    # Do some quick QC to remove bizarre values and save to netCDF file
    output[output<0] = -999
    if (var == 'temp'):
        out_temps[:,:] = output

    elif (var == 'rh'):
        output[output>101] = -999
        out_rh[:,:] = output

    elif (var == 'vapor'):
        output[output>50.0] = -999
        out_vapor[:,:] = output
    
    elif (var == 'liquid'):
        output[output>10.0] = -999
        out_liquid[:,:] = output

pblh_var[:] = tools.compute_pblh(np.array(out_temps), np.array(out_vapor), np.array(radiometer_heights))

# Close the netCDF4 file
fn.close()