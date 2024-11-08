### This module contains the functions for plotting observations from the Valpo University radiometer.
### 
### Christopher Phillips
### github: sodoesaburningbus
### email: christopher.phillips1@valpo.edu

### import required modules
from datetime import datetime, timedelta
import matplotlib.pyplot as pp
import netCDF4 as nc
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

### Function to create the 3-panel plot
### using processed data
def plot_3panel(nc_file, spath, ztop=None, fs=14, fw='bold'):

    # Open the file and read in the data
    fn = nc.Dataset(nc_file, 'r')

    temp = fn.variables['TEMP'][:]
    temp_unit = fn.variables['TEMP'].units
    rh = fn.variables['RH'][:]
    rh_unit = fn.variables['RH'].units
    vapor = fn.variables['VAPOR'][:]
    vapor_unit = fn.variables['VAPOR'].units

    times = fn.variables['time'][:]
    heights = fn.variables['height'][:]
    date = datetime.strptime(fn.date_created, "%Y-%m-%d %H:%M UTC")

    # Nan out any data above the desired model top
    if (ztop != None):
        zbound = np.nanargmin((heights-ztop)**2)+2 # Use two heights up to keep plot looking nice
        temp[:,zbound:] = np.nan
        rh[:,zbound:] = np.nan
        vapor[:,zbound:] = np.nan

    # Make the plot
    fig, axes = pp.subplots(nrows=3, constrained_layout=True, figsize=(9, 6.5))
    fig.suptitle(f'Valparaiso Radiometer {date.strftime("%Y-%m-%d")}', fontsize=fs, fontweight=fw)

    # Temperature
    tcon = axes[0].pcolormesh(times, heights, temp.transpose(), cmap='plasma', shading='nearest', vmin=np.floor(np.nanmin(temp)), vmax=np.ceil(np.nanmax(temp)))
    tcb = fig.colorbar(tcon, ax=axes[0], orientation='vertical')
    tcb.set_label(f'Temperature ({temp_unit})', fontsize=fs, fontweight=fw)

    dT = np.floor((np.ceil(np.nanmax(temp))-np.floor(np.nanmin(temp)))/6.0)
    Tlevs = np.arange(np.floor(np.nanmin(temp)), np.ceil(np.nanmax(temp)), dT, dtype='int')
    tcb.set_ticks(Tlevs)

    axes[0].set_xlim(0,24)
    axes[0].set_title('Temperature', fontsize=fs, fontweight=fw, loc='left', ha='left')

    # Relative Humidity
    rcon = axes[1].pcolormesh(times, heights, rh.transpose(), cmap='YlGnBu', vmin=0, vmax=100, shading='nearest')
    rcb = fig.colorbar(rcon, ax=axes[1], orientation='vertical')
    rcb.set_label(f'Relative\nHumidity ({rh_unit})', fontsize=fs, fontweight=fw)

    axes[1].set_xlim(0,24)
    axes[1].set_title('Relative Humidity', fontsize=fs, fontweight=fw, loc='left', ha='left')

    # Vapor Density
    rcon = axes[2].pcolormesh(times, heights, vapor.transpose(), cmap='YlGnBu', shading='nearest', vmin=np.floor(np.nanmin(vapor)), vmax=np.ceil(np.nanmax(vapor)))
    rcb = fig.colorbar(rcon, ax=axes[2], orientation='vertical')
    rcb.set_label(f'Vapor\nDensity ({vapor_unit})', fontsize=fs, fontweight=fw)

    dV = np.ceil((np.ceil(np.nanmax(vapor))-np.floor(np.nanmin(vapor)))/6.0)
    rcb.set_ticks(np.arange(np.floor(np.nanmin(vapor)), np.ceil(np.nanmax(vapor))+dV, dV, dtype='int'))

    axes[2].set_xlim(0,24)
    axes[2].set_title('Vapor Density', fontsize=fs, fontweight=fw, loc='left', ha='left')
    axes[2].set_xlabel('Hour from Midnight (UTC)', fontsize=fs, fontweight=fw)

    for ax in axes.flatten():
        ax.grid()
        ax.set_ylabel('Height (km)', fontsize=fs, fontweight=fw)

    # Adjust y limit if desired
    if (ztop != None):
        for ax in axes.flatten():
            ax.set_ylim(0, ztop)

    # Save the figure
    pp.savefig(spath)
    pp.close()

    return