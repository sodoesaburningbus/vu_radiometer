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
sdir = 'training_data/hotfix/ '

#####  END OPTIONS  #####

### Import required modules
from src import readers
from src import tools.filter_level_1 as filter

### Locate the and loop over the level 1 and level 2 files
level1_files = sorted(glob(f'{level1_dir}/*.lv1.csv'))
level2_files = sorted(glob(f'{level2_dir}/*.lv2.csv'))

# Check that length of the lists match
if (len(level1_files) != len(level2_files))
    raise Exception('ERROR! Mismatchedd number of Level 1 and Level 2 data')

# Loop over the data files

    # Read in and filter the level 1 data
    
    # Read in the level 2 data and split into temperature, vapor and liquid mixing ratios, and humidity

    # Apply filters to level 2 data

    # Store level 1 brightness temperature as inputs

    # Store level 2 variables as targets (each one separately)

### Convert everything to arrays and store as a netCDF file