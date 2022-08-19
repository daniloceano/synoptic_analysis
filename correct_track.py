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
from metpy.constants import g
from metpy.units import units
import time

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
    
def plot_tracks():
    track = pd.read_csv('./track',sep= ';',index_col=0,header=0)
    for i in range(3):
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
        # plot track
        if i == 0 or i == 1:
            ax.plot(track['Lon'],track['Lat'],c='#BF3D3B',
                    label='track',linewidth=2)
            ax.scatter(track['Lon'],track['Lat'],s=50,c='#BF3D3B',
                       edgecolor='k')
        # plot first_guess
        if i == 0 or i == 2:
            ax.plot(first_guess['Lon'],first_guess['Lat'],c='#383838',
                    label='first guess')
            ax.scatter(first_guess['Lon'],first_guess['Lat'],edgecolor='k',
                       linewidth=2,s=50)
        # decorators
        map_features(ax)
        if i == 0:
            plt.legend(fontsize=18)
            plt.savefig(outdir+'track_compare',bbox_inches='tight')
        elif i == 1:
            plt.title('Track', fontsize=18)
            plt.savefig(outdir+'track',bbox_inches='tight')
        elif i == 2:
            plt.title('First guess', fontsize=18)
            plt.savefig(outdir+'track_first_guess',bbox_inches='tight')

def plot_intensity():
    # coordinate/variable names
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    # model level closest to ground
    sfc_lvl = np.amax(NetCDF_data[LevelIndexer])
    timesteps = NetCDF_data[TimeIndexer]
    # Get Ght value for each track point
    track = pd.read_csv('./track',sep= ';',index_col=0,header=0)
    track_ghts = []
    fg_ghts = []
    for t in timesteps:
        itime = pd.to_datetime(t.values).strftime('%Y-%m-%d-%H%M')
        idata = NetCDF_data.sel({TimeIndexer:t}).sel(
            {LevelIndexer:sfc_lvl})
        if args.geopotential:
            ght = (idata[dfVars.loc['Geopotential']['Variable']] \
            * units(dfVars.loc['Geopotential']['Units'])/g).metpy.convert_units('gpm')
        else:
            ght = idata[dfVars.loc['Geopotential Height']['Variable']]\
                *units(dfVars.loc['Geopotential Height']['Units']).to('gpm')
                
        # Get closest grid point to track 
        track_lon = track.loc[itime]['Lon']
        track_lat = track.loc[itime]['Lat']
        x = float((idata[LonIndexer]
                              [(np.abs(idata[LonIndexer] - 
                              track_lon)).argmin()]))
        y = float((idata[LonIndexer]
                              [(np.abs(idata[LonIndexer] - 
                              track_lat)).argmin()]))
        
        track_ghts.append(float(ght.sel({LonIndexer:x,LatIndexer:y})))
        
        # Get closest grid point to first guess 
        fg_lon = first_guess.loc[itime]['Lon']
        fg_lat = first_guess.loc[itime]['Lat']
        x = float((idata[LonIndexer]
                              [(np.abs(idata[LonIndexer] - 
                              fg_lon)).argmin()]))
        y = float((idata[LonIndexer]
                              [(np.abs(idata[LonIndexer] - 
                              fg_lat)).argmin()]))
        
        fg_ghts.append(float(ght.sel({LonIndexer:x,LatIndexer:y})))
    
    for i in range(3):
        plt.close('all')
        plt.figure(constrained_layout=False,figsize=(12,10))
        ax = plt.gca()
        
        if i == 0 or i == 1:
            ax.plot(timesteps.values,track_ghts,c='#BF3D3B',
                    label='track',linewidth=2)
            ax.scatter(timesteps.values,track_ghts,s=50,c='#BF3D3B',
                       edgecolor='k')
        # plot first_guess
        if i == 0 or i == 2:
            ax.plot(timesteps.values,fg_ghts,c='#383838',
                    label='first guess')
            ax.scatter(timesteps.values,fg_ghts,edgecolor='k',
                       linewidth=2,s=50)
        plt.grid(c='gray',linewidth=0.25,linestyle='dashdot')
        plt.tick_params(axis='x', labelrotation=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.xlim([timesteps[0].values,timesteps[-1].values])
        if i == 0:
            plt.legend(fontsize=18)
            plt.savefig(outdir+'intensity_compare',bbox_inches='tight')
        elif i == 1:
            plt.title('Track', fontsize=18)
            plt.savefig(outdir+'intensity_track',bbox_inches='tight')
        elif i == 2:
            plt.title('First guess', fontsize=18)
            plt.savefig(outdir+'intensity_first_guess',bbox_inches='tight')

def main(): 

    # model level closest to ground
    sfc_lvl = np.amax(NetCDF_data[LevelIndexer])
    # loop through time steps
    timesteps = NetCDF_data[TimeIndexer]
    mins = []
    track_dict = {'time':[],'Lat':[],'Lon':[]}
    for t in timesteps:
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
        if args.geopotential:
            ght = (idata[dfVars.loc['Geopotential']['Variable']] \
           * units(dfVars.loc['Geopotential']['Units'])/g).metpy.convert_units('gpm')
        else:
            ght = idata[dfVars.loc['Geopotential Height']['Variable']]\
                *units(dfVars.loc['Geopotential Height']['Units']).to('gpm')
        # find minimum ght
        min_ght = ght.min()
        mins.append(float(min_ght))
        # Location of the mininum ght as guessed by the user
        first_guess_loc = [first_guess.loc[itime]['Lon'],
                           first_guess.loc[itime]['Lat']]
        
        # get location of the mininum ght tracked
        min_ght_loc = ght.where(ght==min_ght, drop=True).squeeze()
        # if can't decide where is minimum data, use first_guess´
        try:
            track_loc = [float(min_ght_loc[LonIndexer]),
                     float(min_ght_loc[LatIndexer])]
        except:
            track_loc = first_guess_loc
        
        track_dict['time'].append(itime)
        track_dict['Lon'].append(track_loc[0])
        track_dict['Lat'].append(track_loc[1])
        
        plot_map(ght,first_guess_loc,track_loc)
    
    # Convert track dict to DataFrame for saving as CSV file
    track = pd.DataFrame(track_dict)
    track.to_csv('track',sep=';',index=False)
    print('track file created!')
    plot_tracks()
    print('created maps with tracks')
    plot_intensity()
    print('created plots with cyclone intensity')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "\
Creates a track file for a transient system from a first_guess file created \
by the user and from a input netCDF file containing geopotential height data. \
A fvars file contaning the naming convenction for the netCDF file\
is required")
    parser.add_argument("infile", help = "input .nc file geopotential height \
data, in pressure levels")
    parser.add_argument("box_size", help = "set the lenght and height, \
in degrees, for the box used for searching the minimum geopotential height. \
It is advised to be set to correspond to 2 model grid points")
    parser.add_argument("-g", "--geopotential", default = False,
    action='store_true', help = "use the geopotential data instead of\
 geopotential height. The file fvars must be adjusted for doing so.")
     # arguments
    args = parser.parse_args()
    infile  = args.infile
    offset_box = float(args.box_size)
    
    # coordinate/variable names
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    # open file
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
    offset_map = 10
    westernlimit = first_guess['Lon'].min()-offset_map
    easternlimit = first_guess['Lon'].max()+offset_map
    southernlimit = first_guess['Lat'].min()-offset_map
    northernlimit = first_guess['Lat'].max()+offset_map
    
    # Data variables
    LonIndexer = dfVars.loc['Longitude']['Variable']
    LatIndexer = dfVars.loc['Latitude']['Variable']
    TimeIndexer = dfVars.loc['Time']['Variable']
    LevelIndexer = dfVars.loc['Vertical Level']['Variable']
    
    # Slice NetCDF file for only the times present in the first_guess file
    NetCDF_data = NetCDF_data.sel({TimeIndexer:
                        slice(first_guess.index[0],first_guess.index[-1])})
    
    # Rune and time the execution
    start_time = time.time()
    main()
    print("--- %s seconds running correct_track ---" % (time.time() - start_time))
