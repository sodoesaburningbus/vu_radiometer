### This module contains various tools for handling the Level 1 and Level 2 data
### Christopher Phillips

### Import modules
import numpy as np

### Function to filter outliers from the Level 1 data
### Outliers are determined using a standard deviation envelope around the mean.
### The outliers are then replaced with the average of the surrounding points.
### Inputs:
###   data, numpy array of floats, data array from Level 1 reader
###   sigma, optional, standard dev. bounds to use as outlier filter. Default=2.5
def filter_level_1(data, sigma=2.5):

    # Copy the data to a new array first
    filtered_data = data.copy()

    # Loop over the radiometer bands
    qc_list = [] # List to hold QC flags for each band
    for i in range(9, 44):

        # Compute standard deviation and mean of this band
        stdev = np.nanstd(data[:,i])
        mean = np.nanmean(data[:,i])

        # Perform the outlier check
        qc = np.abs(data[:,i]-mean)>(sigma*stdev)
        qc_list.append(qc)

        # Remove outliers
        # Interior elements
        for j in range(1, qc.size-1):
            if qc[j]:
                filtered_data[j] = (filtered_data[j-1]+filtered_data[j+1])/2.0

        # Exterior elements
        if qc[0]:
            filtered_data[0] = filtered_data[1]
        if qc[-1]:
            filtered_data[-1] = filtered_data[-2]

    # Convert list of QC for each band to a numpy array
    qc_list = np.array(qc_list)

    return filtered_data, qc_list