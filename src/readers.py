### This module contains the functions for reading the Valpo University radiometer.
### Note that only the zenith observations are read.
### 
### Christopher Phillips
### github: sodoesaburningbus
### email: christopher.phillips1@valpo.edu

### Import modules
import numpy as np
from datetime import datetime

### Function to read the Level 1 files
### Inputs:
###   filepath, string, path to CSV file
###
### Outputs:
###   data, numpy array of floats, column order is:
###     hours since midnight, Tamb(K), RH(%), Pres(mb), Tir(K), Rain flag, QCamb flag, Elev (deg), Tbb(K), CH1 (K), ..., CH35 (K), QCir
###     The IR data points (Elev and after) repeat for each elevation angle
def read_level_1(filepath):

    # Open file and read line by line
    fn = open(filepath)
    data = [] # For storing all data
    subdata = [] # For storing each record
    for line in fn:

        # Split line on commas
        split_line = line.split(',')

        # Skip if header
        if (split_line[2] not in ['41','51']):
            continue

        # Determine if ambient meteorology or brightness temperature row
        elif (split_line[2] == '41'): # Ambient meteorology line

            # Check that no record already exists
            if (len(subdata) > 0):
                data.append(subdata)
                subdata = []

            # Process time
            subdata += [(datetime.strptime(split_line[1],"%m/%d/%y %H:%M:%S")-datetime.strptime(split_line[1][:8],"%m/%d/%y")).total_seconds()/3600.0]
            
            # Store other data
            subdata += split_line[3:]

        elif (split_line[2] == '51'): # IR brightness temperatures
            if (abs(float(split_line[4])-90.0) <= 2.5): # only add if within 2.5 degrees of zenith
                subdata += split_line[4:]

    # Close data file
    fn.close()

    # Handle last row
    if (len(subdata) == len(data[-1])): # Check that data record is consistent and file not ending abruptly
        data.append(subdata)

    # Convert data list to numpy array
    data = np.array(data, dtype='float')

    return data

### Function to read the Level 2 files
### Inputs:
###   filepath, string, path to CSV file
###
### Outputs:
###   data, numpy array of floats, column order is:
###     hours since midnight, Tamb(K), RH(%), Pres(mb), Tir(K), Rain flag, QCamb flag, Elev (deg), Z1, ..., Z35, QCz, Integrated Vapor (cm), Integrated Liquid (mm), Cloud Base (km), QCint
###     Note that the height channels contain, Temperature (K), Vapor (g/m3), Liquid (g/m3), and Humidity (%) in that order.
###     Thus, the data is arranged, ambient, the four height channels, and then integrated values.
###
###   heights, numpy array of floats, these are the heights (km) of the retrievals
###   
def read_level_2(filepath):

     # Open file and read line by line
    fn = open(filepath)
    data = [] # For storing all data
    subdata = [] # For storing each record
    first_pass = True # Flag for if first pass through the metadata
    for line in fn:

        # Split line on commas
        split_line = line.split(',')

        # Skip if header
        if (split_line[2] not in ['31', '201','301', '400', '401', '402', '403', '404']):
            continue

        elif (split_line[2] == '400'): # Grab the heights
            heights = np.array(split_line[4:-1], dtype='float')

        elif (first_pass and (split_line[2] == '31')): # Add radiometer height to the height array, but only once.
            heights += float(split_line[10])/1000.0
            first_pass = False

        elif (split_line[2] == '201'):

            # Check that no record already exists
            if (len(subdata) > 0):
                data.append(subdata)
                subdata = []

            # Process time
            subdata += [(datetime.strptime(split_line[1],"%m/%d/%y %H:%M:%S")-datetime.strptime(split_line[1][:8],"%m/%d/%y")).total_seconds()/3600.0]
            
            # Store other data
            subdata += split_line[3:]

        elif (split_line[2] in ['401', '402', '403', '404']):

            # Track the elevation angle for checks
            elevation = split_line[3]

            # Add the data only if Zenith
            if (split_line[3] == 'Zenith'):
                subdata += [90.0]
                subdata += split_line[4:]

        elif (split_line[2] == '301'):

            # Check that this is a set of zenith observations
            if (elevation == 'Zenith'):
                subdata += split_line[3:]

    # Close data file
    fn.close()

    # Handle last row
    if (len(subdata) == len(data[-1])): # Check that data record is consistent and file not ending abruptly
        data.append(subdata)

    # Convert data list to numpy array
    data = np.array(data, dtype='float')

    return data, heights