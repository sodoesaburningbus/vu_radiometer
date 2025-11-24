### This module contains the functions for plotting observations from the Valpo University radiometer.
### 
### Christopher Phillips
### github: sodoesaburningbus
### email: christopher.phillips1@valpo.edu

### import required modules
from datetime import datetime, timedelta
import matplotlib.pyplot as pp
import metpy.calc as mcalc
from metpy.plots import SkewT
from metpy.units import units as mu
import netCDF4 as nc
import numpy as np
import pytz


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
    pblh = fn.variables['PBLH'][:]

    times = fn.variables['time'][:]
    heights = fn.variables['height'][:]
    date = datetime.strptime(fn.date_created, "%Y-%m-%d %H:%M UTC")

    fn.close()

    # Check if DST for time zone conversion
    # Will only convert tick labels. Actual times remain UTC to match file handling.
    localtime = pytz.timezone('US/Central')
    if (localtime.localize(date).dst()):
        time_shift = -5
    else:
        time_shift = -6

    # Nan out any data above the desired model top
    if (ztop != None):
        zbound = np.nanargmin((heights-ztop)**2)+2 # Use two heights up to keep plot looking nice
        temp[:,zbound:] = np.nan
        rh[:,zbound:] = np.nan
        vapor[:,zbound:] = np.nan

    # Make the plot
    fig, axes = pp.subplots(nrows=3, constrained_layout=True, figsize=(9, 6.5))
    fig.suptitle(f'Valparaiso Radiometer {date.strftime("%b %d, %Y")}', fontsize=fs, fontweight=fw)

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
    axes[2].set_xlabel('Time (Local)', fontsize=fs, fontweight=fw)

    for ax in axes.flatten():
        ax.grid()
        ax.set_ylabel('Height (km)', fontsize=fs, fontweight=fw)
        ax.set_xticks(np.arange(0, 26, 2, dtype='int'))
        ax.set_xticklabels((np.arange(0, 26, 2, dtype='int')+time_shift)%24)
        ax.plot(times, pblh, linestyle='--', linewidth=1.2, color='black')

    # Adjust y limit if desired
    if (ztop != None):
        for ax in axes.flatten():
            ax.set_ylim(0, ztop)

    # Save the figure
    pp.savefig(spath)
    pp.close()

    return

# Plot the Skew-T from the most recent time.
# Uses processed data.
def plot_skewt(nc_file, spath, ztop=None, fs=14, fw='bold'):

    # Read in the data
    # Open the file and read in the data
    fn = nc.Dataset(nc_file, 'r')

    temp = fn.variables['TEMP'][-1,:]
    rh = fn.variables['RH'][-1,:]
    vapor = fn.variables['VAPOR'][-1,:]
    pblh = fn.variables['PBLH'][-1]

    times = fn.variables['time'][-1]
    heights = fn.variables['height'][:]
    t0 = fn.variables['Tamb'][-1]
    p0 = fn.variables['Pamb'][-1]*100.0 # hPa -> Pa
    date = datetime.strptime(fn.date_created, "%Y-%m-%d %H:%M UTC")

    fn.close()

    # Get the local date
    localtime = pytz.timezone('US/Central')
    if (localtime.localize(date).dst()):
        time_shift = -5
    else:
        time_shift = -6
    local_date = date+timedelta(hours=time_shift)

    # Compute the pressure using the hypsometric equation
    # First assume dry atmosphere, then compute virtual temperature.
    # Then iterate

    # Dry atmosphere
    tbar = (temp[1:]+temp[:-1])*0.5
    hdif = (heights[1:]-heights[:-1])*1000.0 # km -> m
    pres = [p0]
    for tb, hd in zip(tbar, hdif):
        pres.append(pres[-1]*np.exp((-9.81*hd)/(287.0*tb)))
    pres = np.array(pres)

    # Iterate for final values using moist atmosphere
    thresh = 0.01
    i = 0
    old_pres = np.zeros(pres.shape)
    rhbar = ((rh[1:]+rh[:-1])/2.0)/100.0 # % -> decimal
    while ((abs(old_pres[-1] - pres[-1]) > thresh) and (i < 200)):

        # Reset the pressures
        old_pres = pres

        # Compute virtual temperature
        pbar = (pres[1:]+pres[:-1])/2.0
        e = 611.2*np.exp(17.67*(tbar-273.15)/((tbar-273.15)+243.5))*rhbar
        w = (287.047/461.5*e)/(pbar-e)
        tvbar = (1.0+(287.047/461.5-1.0)*(w/(1.0+w)))*tbar

        # Compute pressure levels
        pres = [p0]
        for tv, hd in zip(tvbar, hdif):
            pres.append(pres[-1]*np.exp((-9.81*hd)/(287.0*tv)))
        pres = np.array(pres)

        i += 1

    #print(i)
    #for p, h in zip(pres/100.0, heights):
    #    print(f"{p:.0f}, {h:.2f}")

    # Prep variables with units
    e = 611.2*np.exp(17.67*(temp-273.15)/((temp-273.15)+243.5))*rh/100.0
    dewp = mu.degC*np.array((-243.5*np.log(e/611.2)/(np.log(e/611.2)-17.67)))
    temp = mu.degC*(temp-273.15)
    pres = mu.hPa*(pres/100.0)
    
    # Compute the derived units
    p_lcl, t_lcl = mcalc.lcl(pres[0], temp[0], dewp[0])
    parcel = mcalc.parcel_profile(pres, temp[0], dewp[0])
    pwat = mcalc.precipitable_water(pres, dewp)
    cape, cin = mcalc.surface_based_cape_cin(pres, temp, dewp)

    # Now make the SkewT
    fig = pp.figure(figsize=(8,5), dpi=300, constrained_layout=True)
    skewt = SkewT(fig, rotation=45)
    skewt.plot(pres, temp, 'firebrick')
    skewt.plot(pres, dewp, 'forestgreen')

    skewt.plot(pres, parcel, color='black')
    skewt.ax.axhline(p_lcl, linestyle='--', color='dodgerblue')
    skewt.ax.axhline(pres[np.argmin((pblh-heights)**2)], linestyle=':', color='black')

    # Set axis limits
    if (ztop != None):
        ptop = heights[np.argmin((heights-ztop)**2)]
        skewt.ax.set_ylim(1000,ptop)
    else:
        ptop = pres[-1]
        skewt.ax.set_ylim(1000, ptop)

    # Titles and text
    skewt.ax.set_title(f'Current Sounding\nVU Radiometer - {local_date.strftime("%b %d, %Y %H:%M Local")}', fontsize=fs+2, fontweight=fw, ha='left', loc='left')
    skewt.ax.set_xlabel('Temperature (°C)', fontweight=fw, fontsize=fs)
    skewt.ax.set_ylabel('Pressure (hPa)', fontweight=fw, fontsize=fs)
    skewt.ax.text(20*mu.degC, p_lcl-10*mu.hPa, 'LCL', color='dodgerblue', fontsize=fs, ha='left', va='bottom', zorder=10)
    skewt.ax.text(25*mu.degC, pres[np.argmin((pblh-heights)**2)]-10*mu.hPa, 'PBLH', color='black', fontsize=fs, ha='left', va='bottom', zorder=9)

    # Create info box
    info = f"CAPE {cape.magnitude:.0f} J/kg\nCIN {cin.magnitude:.0f} J/kg\nPWAT {pwat.magnitude:.0f} mm"
    skewt.ax.text(-20*mu.degC, ptop+50*mu.hPa, info, color='black', fontsize=fs-2, ha='left', va='top', zorder=11,
        bbox=dict(edgecolor='black', boxstyle='round,pad=1', facecolor='white', alpha=0.8))

    pp.savefig(spath)
    pp.close()

    return