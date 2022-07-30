#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:58:31 2022

Created by:
    Danilo Couto de Souza
    Universidade de São Paulo (USP)
    Instituto de Astornomia, Ciências Atmosféricas e Geociências
    São Paulo - Brazil

Contact:
    danilo.oceano@gmail.com
"""

import pandas as pd
import xarray as xr
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import cartopy.crs as ccrs
from matplotlib import cm
from main import convert_lon
from plot_maps import Brazil_states, map_features, grid_labels_params

"""

TO DO: use argparse!!!!!! make timeseries 


"""
def plot_map(ght,first_guess_loc,track_loc):
    # projection
    proj = ccrs.PlateCarree() 
    # create figure
    plt.close('all')
    fig = plt.figure(constrained_layout=False,figsize=(12,10))
    # create subplot
    ax = fig.add_subplot(projection=proj)
    ax.set_extent([westernlimit,easternlimit,
                   southernlimit,northernlimit]) 
    # Add decorators and Brazil states
    grid_labels_params(ax,0)
    Brazil_states(ax)
    # get latitude and longitude
    lon,lat = ght[LonIndexer], ght[LatIndexer]
    # get data range
    max1,min1 = float(np.amax(ght)),float(np.amin(ght))
    cmap = cmo.balance
    norm = cm.colors.Normalize(vmax=max1*1.01,vmin=min1*.95)
    # plot shaded
    cf1 = ax.contourf(lon, lat, ght, cmap=cmap,norm=norm) 
    ax.contour(lon, lat, ght,cf1.levels,colors='#383838',
           linewidths=0.2)
    # plot the selected box around the contours
    min_lon = float(ght[LonIndexer].min())
    max_lon = float(ght[LonIndexer].max())
    min_lat = float(ght[LatIndexer].min())
    max_lat = float(ght[LatIndexer].max())
    ax.plot([min_lon,min_lon,max_lon,max_lon,min_lon],
            [min_lat,max_lat,max_lat,min_lat,min_lat],
            linewidth=1,c='#383838',zorder=500)
    #  plot the system center
    ax.scatter(first_guess_loc[0],first_guess_loc[1], s=50,
               zorder=501, color='#383838',linewidth=2,edgecolor='k')
    ax.scatter(track_loc[0],track_loc[1], s = 100,
               zorder=502, color='r',linewidth=2,edgecolor='k')
    # get time string
    timestr = pd.to_datetime(str(ght[TimeIndexer].values))
    date = timestr.strftime('%Y-%m-%dT%H%MZ')
    # Title
    title = 'Geopot. height at 1000 hPa (color contours) in the '\
        'box used for the tracking, \n'\
        'system center (dots, first_guess: black, track: red)'\
        ', date: '+str(date)
    ax.text(0,1.01,title, transform=ax.transAxes, fontsize=16)
    # colorbar
    cbar = plt.colorbar(cf1,fraction=0.046, pad=0.07, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10) 
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(10)
    # decorators
    map_features(ax)
    # save file
    outfile = FigsDirectory+'map_ght_'+str(date)
    plt.savefig(outfile,bbox_inches='tight')
    print(outfile+' created!')
    
#------------------------------------------------------

infile = '../lorenz_etc/Reg1-Yakecan_NCEP-R2.nc'
# The size of the box in degrees
offset_box = 5
dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
NetCDF_data = convert_lon(xr.open_dataset(infile),
                          dfVars.loc['Longitude']['Variable'])

# Directories
output = infile.split('/')[-1].split('.')[0]
outdir = "../SynopticAnalysis_Results/"+output+"/"; os.makedirs(
    outdir, exist_ok=True)
FigsDirectory = outdir+'compare_tracks/'; os.makedirs(
    FigsDirectory, exist_ok=True)

# File with tracks defined 'by the eye'
first_guess =  pd.read_csv('./first_guess',sep= ';',index_col=0,header=0)

# Limits for plotting (bigger plot)
westernlimit = first_guess['Lon'].min()-2
easternlimit = first_guess['Lon'].max()+2
southernlimit = first_guess['Lat'].min()-2
northernlimit = first_guess['Lat'].max()+2

# Data variables
LonIndexer = dfVars.loc['Longitude']['Variable']
LatIndexer = dfVars.loc['Latitude']['Variable']
TimeIndexer = dfVars.loc['Time']['Variable']
LevelIndexer = dfVars.loc['Vertical Level']['Variable']

# model level closest to ground
sfc_lvl = np.amax(NetCDF_data[LevelIndexer])
# loop through time steps
timesteps = NetCDF_data[TimeIndexer]
mins = []
for t in timesteps[:2]:
    # Get current time and time strings
    itime = pd.to_datetime(t.values).strftime('%Y-%m-%d-%H%M')
    # Get current time and box limits
    min_lon = first_guess.loc[itime]['Lon']-offset_box
    max_lon = first_guess.loc[itime]['Lon']+offset_box
    min_lat = first_guess.loc[itime]['Lat']-offset_box
    max_lat = first_guess.loc[itime]['Lat']+offset_box
    print('\nComputing terms for '+itime+'...')
    print('Box limits (lon/lat): '+str(max_lon)+'/'+str(max_lat),
          ' '+str(min_lon)+'/'+str(min_lat))
    # Get closest grid point to actual track limits
    WesternLimit = float((NetCDF_data[LonIndexer]
                         [(np.abs(NetCDF_data[LonIndexer] - 
                          min_lon)).argmin()]))
    EasternLimit = float((NetCDF_data[LonIndexer]
                          [(np.abs(NetCDF_data[LonIndexer] - 
                          max_lon)).argmin()]).values)
    SouthernLimit = float((NetCDF_data[LatIndexer]
                           [(np.abs(NetCDF_data[LatIndexer] - 
                           min_lat)).argmin()]).values)
    NorthernLimit = float((NetCDF_data[LatIndexer]
                           [(np.abs(NetCDF_data[LatIndexer] - 
                           max_lat)).argmin()]).values)
    # Slice data for surface
    idata = NetCDF_data.sel({TimeIndexer:t}).sel(
        {LevelIndexer:sfc_lvl}).sel(
        **{LatIndexer:slice(NorthernLimit,SouthernLimit),
           LonIndexer: slice(WesternLimit,EasternLimit)})
    ght = idata[dfVars.loc['Geopotential Height']['Variable']]
    # find minimum ght
    min_ght = ght.min()
    mins.append(float(min_ght))
    # Location of the mininum ght as guessed by the user
    first_guess_loc = [first_guess.loc[itime]['Lon'],
                       first_guess.loc[itime]['Lat']]
    
    # get location of the mininum ght tracked
    min_ght_loc = ght.where(ght==min_ght, drop=True).squeeze()
    # if can decide where is minimum data, use first_guess´
    try:
        track_loc = [float(min_ght_loc[LonIndexer]),
                 float(min_ght_loc[LatIndexer])]
    except:
        track_loc = min_ght_loc
    
    plot_map(ght,first_guess_loc,track_loc)
