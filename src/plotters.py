### This module contains the functions for plotting observations from the Valpo University radiometer.
### 
### Christopher Phillips
### github: sodoesaburningbus
### email: christopher.phillips1@valpo.edu

### import required modules
from datetime import datetime, timedelta
import matplotlib.pyplot as pp
import numpy as np

### Function to plot timeseries of each channels brightness temperature
### Inputs:
###   data, numpy array of floats, observations from level 1 reader
###   filepath, string, path to radiometer input file
def plot_bands(data, filepath):

    # Create the figure
    fig, ax = pp.subplots(figsize=(9,6.5), constrained_layout=True)

    # Add each band
    for i in range(9, 44):
        ax.plot(data[:,i], label=f'Band {i-8}')

    ax.legend()

    pp.show()
    pp.savefig(filepath.replace('.csv', '_temperature.png'))
    pp.close()

    return

### Function to plot radiometer temperature
### Inputs:
###   data, numpy array of floats, observations from level 2 reader
###   heights, numpy array of floats, heights from level 2 reader
###   filepath, string, path to radiometer input file
def plot_temperature(data, heights, filepath, zmax=10):

    # Grab the data from the filename
    date = filepath.split('/')[-1][:10]

    # Extract the values of interest
    temps = data[:,8:66] # Temperatures (K)
    qc = data[:,66]
    times = data[:,0]
    
    # Limit data to just the desired height
    for i in range(heights.size):
        if (heights[i] > zmax):
            temps[:,i] = np.nan

    # Apply QC check
    for i in range(qc.size):
        if (qc[i] != 1):
            temps[i,:] = np.nan

    # Make the plot
    fig, ax = pp.subplots(figsize=(9,6.5), constrained_layout=True)

    cont = ax.contourf(times, heights, temps.transpose(), cmap='turbo')
    ax.set_title(f'Valparaiso Radiometer - {date}', fontsize=14, fontweight='roman', loc='left', ha='left')
    ax.set_xlabel(f'Hours since Midnight (UTC)', fontsize=14, fontweight='roman')
    ax.set_ylabel('Height (km)', fontsize=14, fontweight='roman')

    cb = fig.colorbar(cont, ax=ax, orientation='vertical')
    cb.set_label('Temperature (K)', fontsize=14, fontweight='roman')


    ax.set_ylim(heights[0], zmax)

    pp.savefig(filepath.replace('.csv', '_temperature.png'))
    pp.close()

    return

def plot_humidity(filepath):

    return