#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:52:34 2022

@author: danilocoutodsouza
"""

import matplotlib.pyplot as plt
import cmocean.cm as cmo
import numpy as np 
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature, COASTLINE
from cartopy.feature import BORDERS
import pandas as pd
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter


def map_features(ax):
    ax.add_feature(COASTLINE)
    ax.add_feature(BORDERS,edgecolor='#383838')
    return ax

def Brazil_states(ax):    
    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                                  name='admin_1_states_provinces_lines')
    _ = ax.add_feature(states, edgecolor='#383838')
    
    cities = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                                  name='populated_places')
    _ = ax.add_feature(cities)
    
def grid_labels_params(ax,i):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    if i not in [0,3]:
        gl.left_labels = False
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    ax.spines['geo'].set_edgecolor('#383838')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def plot_levels(ShadedVar,FigsDirectory,fname,ContourVar=None,u=None,v=None):
    """
    Parameters
    ----------
    VariableData : DataObj
        Object containing meteorological data from a NetCDF file
    FigsDirectory : str
        Directory where images will be saved.
    fname : str
        Name to append to outfile.

    Returns
    -------
    Create maps for each time step for model levels closest to 1000,850,700,
    500,300 and 200 hPa.
    """
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    LonIndexer,LatIndexer,TimeIndexer,LevelIndexer = \
      dfVars.loc['Longitude']['Variable'],dfVars.loc['Latitude']['Variable'],\
      dfVars.loc['Time']['Variable'],dfVars.loc['Vertical Level']['Variable']
    dfbox = pd.read_csv('./box_limits',delimiter=';',index_col=0,header=None)
    # limits for the map (represents the maximum limits of the lagrangian box)
    westernlimit = find_nearest(ShadedVar[LonIndexer],
                                float(dfbox.loc['min_lon'].values))
    easternlimit = find_nearest(ShadedVar[LonIndexer],
                                float(dfbox.loc['max_lon'].values))
    southernlimit = find_nearest(ShadedVar[LatIndexer],
                                 float(dfbox.loc['min_lat'].values))
    northernlimit = find_nearest(ShadedVar[LonIndexer],
                                 float(dfbox.loc['max_lat'].values))
    # projection
    proj = ccrs.PlateCarree() 
    # create figure
    plt.close('all')
    fig = plt.figure(constrained_layout=False,figsize=(24,20))
    gs = gridspec.GridSpec(2, 3, hspace=0.1, wspace=0.1,
                                   left=0.05, right=0.95)
    # Find vertical levels that closely matches the desired levels for plotting
    MatchingLevels = []
    for pres in [1000,850,700,500,300,200]:
        MatchingLevels.append(min(ShadedVar[LevelIndexer].values, key=lambda x:abs(x-pres)))
    # loop through pressure levels
    for p,i in zip(MatchingLevels,range(len(MatchingLevels))):
        # Get current time and time strings
        itime = ShadedVar[TimeIndexer].values
        # create subplot
        ax = fig.add_subplot(gs[i], projection=proj)
        ax.set_extent([westernlimit,easternlimit,southernlimit,northernlimit]) 
        # Add decorators and Brazil states
        grid_labels_params(ax,i)
        Brazil_states(ax)
        # Slice data for the desired domain and pressure level
        iShadedVar = ShadedVar.sel({LevelIndexer:p}).sel(
            **{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)})
        if ContourVar is not None:
            iContourVar = ContourVar.sel({LevelIndexer:p}).sel(
                **{LatIndexer:slice(northernlimit,southernlimit),
                   LonIndexer: slice(westernlimit,easternlimit)})
        if u is not None and v is not None:
            iu = u.sel({LevelIndexer:p}).sel(
                **{LatIndexer:slice(northernlimit,southernlimit),
                   LonIndexer: slice(westernlimit,easternlimit)})
            iv = v.sel({LevelIndexer:p}).sel(
                **{LatIndexer:slice(northernlimit,southernlimit),
                   LonIndexer: slice(westernlimit,easternlimit)})
        # get latitude and longitude
        lon,lat = iShadedVar[LonIndexer], iShadedVar[LatIndexer]
        # get data range
        max1,min1 = float(np.amax(iShadedVar)), float(np.amin(iShadedVar))
        # (if data is an annomaly)
        if min1 > 0:
            cmap = cmo.amp
            norm = cm.colors.Normalize(vmax=max1,vmin=min1)
        # (if data is not an annomaly)
        else:
            cmap = cmo.balance
            norm = colors.TwoSlopeNorm(vmin=min1, vcenter=0, vmax=max1)
        # plot shaded
        cf1 = ax.contourf(lon, lat, iShadedVar, cmap=cmap,norm=norm) 
        ax.contour(lon, lat, iShadedVar,cf1.levels,colors='#383838',
               linewidths=0.2)
        # plot contour
        if ContourVar is not None:
            levels = np.linspace(np.amin(iContourVar),np.amax(iContourVar),12)
            cs = ax.contour(lon, lat, iContourVar,levels,colors='#383838',
                   linewidths=2.5)
            ax.clabel(cs, cs.levels, inline=True,fmt = '%1.0f', fontsize=12)
        if u is not None and v is not None:
            ax.barbs(lon, lat, iu.metpy.convert_units('kt'),
                     iv.metpy.convert_units('kt'))
        # get time string
        timestr = pd.to_datetime(str(iShadedVar[TimeIndexer].values))
        date = timestr.strftime('%Y-%m-%dT%H%MZ')
        # Title
        title = iShadedVar.name+' (shaded, '+str(iShadedVar.metpy.units)+')'
        if ContourVar is not None:
            title += ', '+ContourVar.name+\
                '\n(contours, '+str(ContourVar.metpy.units)+')'
        if u is not None and v is not None:
            title += ', wind (barbs, kt)' 
        title += ' for '+str(date)
        
        ax.text(0.01,1.01,str(p)+' '+str(ShadedVar[LevelIndexer].units)+' '+title,
                transform=ax.transAxes, fontsize=16)
        # colorbar
        cbar = plt.colorbar(cf1,fraction=0.046, pad=0.07, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10) 
        for t in cbar.ax.get_yticklabels():
             t.set_fontsize(10)
        # decorators
        map_features(ax)
    # save file
    outfile = FigsDirectory+'/map_'+fname+'_'+str(date)
    plt.savefig(outfile)
    print(outfile+' created!')
    
def plot_ThetaHgtWind(ThetaData,HgtData,u,v,FigsDirectory,fname):
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    LonIndexer,LatIndexer,TimeIndexer,LevelIndexer = \
      dfVars.loc['Longitude']['Variable'],dfVars.loc['Latitude']['Variable'],\
      dfVars.loc['Time']['Variable'],dfVars.loc['Vertical Level']['Variable']
    dfbox = pd.read_csv('./box_limits',delimiter=';',index_col=0,header=None)
    # limits for the map (represents the maximum limits of the lagrangian box)
    westernlimit = find_nearest(ThetaData[LonIndexer],
                                float(dfbox.loc['min_lon'].values))
    easternlimit = find_nearest(ThetaData[LonIndexer],
                                float(dfbox.loc['max_lon'].values))
    southernlimit = find_nearest(ThetaData[LatIndexer],
                                 float(dfbox.loc['min_lat'].values))
    northernlimit = find_nearest(ThetaData[LonIndexer],
                                 float(dfbox.loc['max_lat'].values))
    # get values for 850 hPa
    lev850 = find_nearest(ThetaData[LevelIndexer],850)
    ThetaData = ThetaData.sel({LevelIndexer:lev850})
    HgtData = HgtData.sel({LevelIndexer:lev850})
    u,v = u.sel({LevelIndexer:lev850}),v.sel({LevelIndexer:lev850})
    # projection
    proj = ccrs.PlateCarree() 
    # create figure
    plt.close('all')
    fig = plt.figure(constrained_layout=False,figsize=(12,10))
    # Get current time and time strings
    itime = ThetaData[TimeIndexer].values
    # create subplot
    ax = fig.add_subplot(projection=proj)
    ax.set_extent([westernlimit,easternlimit,southernlimit,northernlimit]) 
    # Add decorators and Brazil states
    grid_labels_params(ax,0)
    Brazil_states(ax)
    # Slice data for the desired domain and pressure level
    ThetaData = ThetaData.sel(**{LatIndexer:slice(
        northernlimit,southernlimit), LonIndexer: slice(
            westernlimit,easternlimit)})
    HgtData = HgtData.sel(**{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)})
    u = u.sel(**{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)})
    v = v.sel( **{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)})
    # get latitude and longitude
    lon,lat = ThetaData[LonIndexer], ThetaData[LatIndexer]
    # get data range
    max1,min1 = float(np.amax(ThetaData)),float(np.amin(ThetaData))
    cmap = cmo.thermal
    norm = cm.colors.Normalize(vmax=max1*1.01,vmin=min1*.95)
    # plot shaded
    cf1 = ax.contourf(lon, lat, ThetaData, cmap=cmap,norm=norm) 
    ax.contour(lon, lat, ThetaData,cf1.levels,colors='#383838',
           linewidths=0.2)
    # plot contour
    levels = np.linspace(np.amin(HgtData),np.amax(HgtData),12)
    cs = ax.contour(lon, lat, HgtData,levels,colors='k',
               linewidths=2.5)
    ax.clabel(cs, cs.levels, inline=True, fmt = '%1.0f', fontsize=12)
    ax.barbs(lon, lat, u.metpy.convert_units('kt'),
                 v.metpy.convert_units('kt'))
    # get time string
    timestr = pd.to_datetime(str(ThetaData[TimeIndexer].values))
    date = timestr.strftime('%Y-%m-%dT%H%MZ')
    # Title
    title = '850 hPa potential temp. (shaded, K), geo. height (contour, m),\n\
wind (barbs, kt) for '+str(date)
    ax.text(0.01,1.01,title, transform=ax.transAxes, fontsize=16)
    # colorbar
    cbar = plt.colorbar(cf1,fraction=0.046, pad=0.07, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10) 
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(10)
    # decorators
    map_features(ax)
    # Figure title
    # save file
    outfile = FigsDirectory+'/map_'+fname+'_'+str(date)
    plt.savefig(outfile)
    print(outfile+' created!')
    
def plot_SLPJetWind(SLPData,u,v,FigsDirectory,fname):
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    LonIndexer,LatIndexer,TimeIndexer,LevelIndexer = \
      dfVars.loc['Longitude']['Variable'],dfVars.loc['Latitude']['Variable'],\
      dfVars.loc['Time']['Variable'],dfVars.loc['Vertical Level']['Variable']
    dfbox = pd.read_csv('./box_limits',delimiter=';',index_col=0,header=None)
    # limits for the map (represents the maximum limits of the lagrangian box)
    westernlimit = find_nearest(SLPData[LonIndexer],
                                float(dfbox.loc['min_lon'].values))
    easternlimit = find_nearest(SLPData[LonIndexer],
                                float(dfbox.loc['max_lon'].values))
    southernlimit = find_nearest(SLPData[LatIndexer],
                                 float(dfbox.loc['min_lat'].values))
    northernlimit = find_nearest(SLPData[LonIndexer],
                                 float(dfbox.loc['max_lat'].values))
    # get wind speed  for 250 hPa
    lev250 = find_nearest(u[LevelIndexer],250)
    u250,v250 = u.sel({LevelIndexer:lev250}),v.sel({LevelIndexer:lev250})
    ws = np.sqrt(u250**2+v250**2)
    # Get surface winds
    lev1000 = find_nearest(u[LevelIndexer],1000)
    u_sfc, v_sfc =  u.sel({LevelIndexer:lev1000}),v.sel({LevelIndexer:lev1000})
    # projection
    proj = ccrs.PlateCarree() 
    # create figure
    plt.close('all')
    fig = plt.figure(constrained_layout=False,figsize=(12,10))
    # Get current time and time strings
    itime = SLPData[TimeIndexer].values
    # create subplot
    ax = fig.add_subplot(projection=proj)
    ax.set_extent([westernlimit,easternlimit,southernlimit,northernlimit]) 
    # Add decorators and Brazil states
    grid_labels_params(ax,0)
    Brazil_states(ax)
    # Slice data for the desired domain and pressure level
    SLPData = SLPData.sel(**{LatIndexer:slice(
        northernlimit,southernlimit), LonIndexer: slice(
            westernlimit,easternlimit)}).metpy.convert_units('hPa')
    ws = ws.sel( **{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)}).metpy.convert_units('m/s')
    u_sfc = u_sfc.sel(**{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)})
    v_sfc = v_sfc.sel(**{LatIndexer:slice(northernlimit,southernlimit),
               LonIndexer: slice(westernlimit,easternlimit)})
    # Remove values lower than 30 m/s from 250 wind
    # ws_masked = np.ma.masked_where(ws.values < 30, ws)
    # get latitude and longitude
    lon,lat = SLPData[LonIndexer], SLPData[LatIndexer]
    # specs
    vmin, vmax = 990, 1040
    cmap = cmo.balance
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=1014, vmax=vmax)
    # plot shaded
    cf1 = ax.contourf(lon,lat,SLPData,cmap=cmap,norm=norm) 
    ax.contour(lon, lat, ws,cf1.levels,colors='#383838',
            linewidths=0.2)    
    # plot contour
    levels = np.linspace(30,110,17,dtype=int)
    cs = ax.contour(lon, lat, ws,levels,colors='k',
               linewidths=2.5)
    ax.clabel(cs, cs.levels, inline=True, fmt = '%1.0f', fontsize=12)
    ax.barbs(lon, lat, u_sfc.metpy.convert_units('kt'),
                 v_sfc.metpy.convert_units('kt'))
    # get time string
    timestr = pd.to_datetime(str(SLPData[TimeIndexer].values))
    date = timestr.strftime('%Y-%m-%dT%H%MZ')
    # Title
    title = '250 hPa wind speed > 30 (contour, m/s), SLP (shaded, hPa),\n\
surface wind (barbs, kt) for '+str(date)
    ax.text(0.01,1.01,title, transform=ax.transAxes, fontsize=16)
    # colorbar
    v = np.linspace(vmin, vmax, 15, endpoint=True,dtype=int)
    cbar = plt.colorbar(cf1,fraction=0.046,pad=0.07,orientation='horizontal',
                        norm=norm,ticks=v)
    cbar.ax.tick_params(labelsize=10) 
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(10)
    # decorators
    map_features(ax)
    # Figure title
    # save file
    outfile = FigsDirectory+'/map_'+fname+'_'+str(date)
    plt.savefig(outfile)
    print(outfile+' created!')