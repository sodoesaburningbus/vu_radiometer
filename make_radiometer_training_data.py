### This program creates the training data
### for converting radiometer brightness temperatures into
### temperature, moisture, and humidity profiles.
### It utilizes level 1 and level 2 radiometer data.
### It is NOT a re-calibration using rawinsondes.
###
### Christopher Phillips
### christopher.phillips1@valpo.edu

##### START OPTIONS #####

# Directory with level 1 data
lev1_dir = '/home/christopher/projects/radiometer/data'

# Directory with level 2 data
lev2_dir = '/home/christopher/projects/radiometer/data'

# Directory in which to save the training data
sdir = '/home/christopher/projects/radiometer/training_data/hotfix'

#####  END OPTIONS  #####

### Import required modules
from src import readers
from src.tools import filter_level_1 as filter

from glob import glob
import numpy as np
import netCDF4 as nc

### Locate the and loop over the level 1 and level 2 files
lev1_files = sorted(glob(f'{lev1_dir}/*_lv1.csv'))
lev2_files = sorted(glob(f'{lev2_dir}/*_lv2.csv'))

# Check that length of the lists match
if (len(lev1_files) != len(lev2_files)):
    raise Exception('ERROR! Mismatched number of Level 1 and Level 2 data')

# Lists to hold the data as it comes in
inputs = []
temp_target = []
vapor_target = []
liquid_target = []
rh_target = []

# Loop over the data files
for f1, f2 in zip(lev1_files, lev2_files):

    # Try to read the data
    try:
        data1 = readers.read_level_1(f1)
        data2, heights = readers.read_level_2(f2)
    except:
        continue

    # Check if data shape 1 is correct
    if (data1.shape[1] != 45):
        continue

    # Check if data shape 2 is correct
    if (data2.shape[1] != 251):
        continue

    # Filter level 1 data
    filtered_data1, qc1 = filter(data1, sigma=2.0)

    # Collapse qc into a 1D array
    qc1 = ~np.any(qc1, axis=0)

    # Split level 2 into temperature, vapor and liquid mixing ratios, and humidity
    temp = data2[:,8:66]
    temp_qc = data2[:,66]
    vapor = data2[:,68:126]
    vapor_qc = data2[:,126]
    liquid = data2[:,128:186]
    liquid_qc = data2[:,186]
    rh = data2[:,188:246]
    rh_qc = data2[:,246]

    # Apply quality checks
    try:
        all_qc = ((temp_qc + vapor_qc + liquid_qc + rh_qc)>0) & qc1
        temp = temp[all_qc]
        vapor = vapor[all_qc]
        liquid = liquid[all_qc]
        rh = rh[all_qc]
        filtered_data1 = filtered_data1[all_qc]
    except: # Sometimes the filtered data is weird it seems
        qc1 = qc1[:-1]
        all_qc = ((temp_qc + vapor_qc + liquid_qc + rh_qc)>0) & qc1
        temp = temp[all_qc]
        vapor = vapor[all_qc]
        liquid = liquid[all_qc]
        rh = rh[all_qc]
        filtered_data1 = filtered_data1[:-1][all_qc]
    
    # Store level 1 brightness temperature as inputs
    inputs.append(filtered_data1)

    # Store level 2 variables as targets (each one separately)
    temp_target.append(temp)
    vapor_target.append(vapor)
    liquid_target.append(liquid)
    rh_target.append(rh)

### Convert everything to arrays and store in numpy binary files
inputs = np.concatenate(inputs, axis=0, dtype=np.float32)
temp_target = np.concatenate(temp_target, axis=0, dtype=np.float32)
vapor_target = np.concatenate(vapor_target, axis=0, dtype=np.float32)
liquid_target = np.concatenate(liquid_target, axis=0, dtype=np.float32)
rh_target = np.concatenate(rh_target, axis=0, dtype=np.float32)

print(inputs.shape)

np.save(f'{sdir}/inputs.npy', inputs)
np.save(f'{sdir}/temp_targets.npy', temp_target)
np.save(f'{sdir}/vapor_targets.npy', vapor_target)
np.save(f'{sdir}/liquid_targets.npy', liquid_target)
np.save(f'{sdir}/rh_targets.npy', rh_target)

print(f'Number of data points found: {inputs.shape[0]}')