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

### This module locates boundary layer height provided radiometer temeprature and water vapor data
def compute_pblh(temp, vapor, heights):

    # Find the first inversion layer from the surface.
    lapse = temp[:,1:]-temp[:,:-1]
    pblh = []
    for i in range(temp.shape[0]):

        # Find height of lowest inversion
        pblh_dummy = heights[np.arange(heights.size, dtype='int')[lapse[i,:]>0][0]]

        # If no inversion exists below 4.5 km (~600 hPa), use the parcel mixing method
        if (pblh_dummy > 4.5):
            Tv = temp[i,:]*(1.0+0.61*vapor[i,:]) # Wrong because vapor is a density not a mixing ratio
            pblh_ind = np.arange(heights.size, dtype='int')[Tv[i,1:]>=Tv[i,0]]
            pblh_dummy = heights[pblh_ind]

        pblh.append(pblh_dummy)

    # Convert PBLH to an array
    pblh = np.array(pblh)

    return pblh

### Function to compute water vapor mixing ratio
def get_mixr(temp, vapor, humidity):

    # First compute the saturation vapor pressure using the Stefan-Boltzmann formula
    Tc = temp-273.15
    es = 611.2*np.exp(17.67*Tc/(Tc+243.5)) # Pa

    # Convert units
    rh = humidity/100.0 # % -> dec
    vd = vapor/1000.0 # g/m3 -> kg/m3

    # Assign some constants
    Rd = 287.05
    Rv = 461.5
    ep = Rd/Rv

    # Quadratic formula
    a = es*rh-Rd*temp/(vd*ep)
    b = es*rh*ep+es*rh-Rd*temp/vd
    c = es*rh*ep

    rv1 = (-b+np.sqrt(b**2-4.0*a*c))/(2.0*a)
    rv2 = (-b-np.sqrt(b**2-4.0*a*c))/(2.0*a)

    # Find the physical solution
    rv = np.where(rv1>0, rv1, rv2)

    return rv

### Function to compute pressure
def get_pressure(temp, mixr, rh):

    # First compute the saturation vapor pressure using the Stefan-Boltzmann formula
    Tc = temp-273.15
    es = 611.2*np.exp(17.67*Tc/(Tc+243.5)) # Pa
    e = es*rh/100.0

    # Assign some constants
    Rd = 287.05
    Rv = 461.5
    ep = Rd/Rv

    pres = ep*e/mixr+e

    return pres