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
lev1_dir = ''

# Directory with level 2 data
lev2_dir = ''

# Directory in which to save the training data
sdir = 'training_data/hotfix/ '

#####  END OPTIONS  #####