### This program creates the plots of the Valparaiso radiometer
### Christopher Phillips

##### START OPTIONS #####

# Directory with the processed data
rdir = '/archive/campus_mesonet_data/rooftop_radiometer/processed_data'

# Directory to which to save images
sdir = '/archive/campus_mesonet_data/rooftop_radiometer/images'

# Font options
fs = 14
fw = 'bold'

#####  END OPTIONS  #####

# Import required modules
from datetime import datetime
from glob import glob
import matplotlib.pyplot as pp
import netCDF4 as nc
import numpy as np

from src import plotters

# Locate the most recent netCDF file and grab the date
date = datetime.utcnow()
nc_file = sorted(glob(f'{rdir}/{date.year}/radiometer_*.nc'))[-1]

# Make the plots
plotters.plot_3panel(nc_file, f'{sdir}/radiometer_short_plot.png', ztop=2.5)
plotters.plot_3panel(nc_file, f'{sdir}/radiometer_tall_plot.png')