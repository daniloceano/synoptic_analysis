#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:58:23 2022


Created by:
    Danilo Couto de Souza
    Universidade de São Paulo (USP)
    Instituto de Astornomia, Ciências Atmosféricas e Geociências
    São Paulo - Brazil

Contact:
    danilo.oceano@gmail.com
"""

from metpy.units import units
from metpy.constants import g
from metpy.constants import Cp_d
from metpy.constants import Re
from metpy.calc import potential_temperature

import pandas as pd
import xarray as xr
import os
import numpy as np
import argparse

from plot_maps import plot_levels,plot_ThetaHgtWind,plot_SLPJetWind

import time

def convert_lon(xr,LonIndexer):
    """
    
    Convert longitudes from 0:360 range to -180:180

    Parameters
    ----------
    xr : xarray.DataArray 
        gridded data.
    LonIndexer : str
        corrdinate indexer used for longitude.

    Returns
    -------
    xr : xarray.DataArray 
        gridded data with longitude converted to desired format.

    """
    xr.coords[LonIndexer] = (xr.coords[LonIndexer] + 180) % 360 - 180
    xr = xr.sortby(xr[LonIndexer])
    return xr

class DataObject:
    """
    Object for storing variables from a NetCDF file on intialization.
    It also computes each term of the Quasi-Geostrophic Equation, except the
    Adiabatic Heating Term (Q) which is estimated as a residual. 
    Note that: Q = J *Cp_d
    """
    def __init__(self,NetCDF_data: xr.Dataset,
                 dfVars: pd.DataFrame,
                 dfbox: pd.DataFrame=None):
        self.LonIndexer = dfVars.loc['Longitude']['Variable']
        self.LatIndexer = dfVars.loc['Latitude']['Variable']
        self.TimeIndexer = dfVars.loc['Time']['Variable']
        self.LevelIndexer = dfVars.loc['Vertical Level']['Variable']
        # When constructing object for eulerian analysis, the data can
        # be sliced beforehand using the dfBox limits, but for the lagrangian
        # analysis (dfBox not specified), we need full data and then it is
        # sliced for each timestep.
        if dfbox is None:
            self.NetCDF_data = NetCDF_data
        else:
            self.WesternLimit = float((NetCDF_data[self.LonIndexer]
                                 [(np.abs(NetCDF_data[self.LonIndexer] - 
                                  float(dfbox.loc['min_lon']))).argmin()]))
            self.EasternLimit = float((NetCDF_data[self.LonIndexer]
                                  [(np.abs(NetCDF_data[self.LonIndexer] - 
                                   float(dfbox.loc['max_lon']))).argmin()]).values)
            self.SouthernLimit = float((NetCDF_data[self.LatIndexer]
                                   [(np.abs(NetCDF_data[self.LatIndexer] - 
                                   float(dfbox.loc['min_lat']))).argmin()]).values)
            self.NorthernLimit = float((NetCDF_data[self.LatIndexer]
                                   [(np.abs(NetCDF_data[self.LatIndexer] - 
                                   float(dfbox.loc['max_lat']))).argmin()]).values)
            self.NetCDF_data = NetCDF_data.sel(
                **{self.LatIndexer:slice(self.NorthernLimit,self.SouthernLimit),
                   self.LonIndexer: slice(self.WesternLimit,self.EasternLimit)})
        self.Temperature = self.NetCDF_data[dfVars.loc['Air Temperature']['Variable']] \
            * units(dfVars.loc['Air Temperature']['Units']).to('K')
        self.u = self.NetCDF_data[dfVars.loc['Eastward Wind Component']['Variable']] \
            * units(dfVars.loc['Eastward Wind Component']['Units']).to('m/s')
        self.v = self.NetCDF_data[dfVars.loc['Northward Wind Component']['Variable']] \
            * units(dfVars.loc['Northward Wind Component']['Units']).to('m/s')
        self.Omega = self.NetCDF_data[dfVars.loc['Omega Velocity']['Variable']] \
            * units(dfVars.loc['Omega Velocity']['Units']).to('Pa/s')
        if args.geopotential:
             self.GeopotHeight = ((self.NetCDF_data[dfVars.loc['Geopotential']['Variable']] \
            * units(dfVars.loc['Geopotential']['Units']))/g).metpy.convert_units('gpm')
        else:
            self.GeopotHeight = self.NetCDF_data[dfVars.loc['Geopotential Height']['Variable']] \
            * units(dfVars.loc['Geopotential Height']['Units']).to('gpm')
        self.Pressure = self.NetCDF_data[self.LevelIndexer]
        try:
            self.SLP = self.NetCDF_data[dfVars.loc['Sea Level Pressure']['Variable']] \
                * units(dfVars.loc['Sea Level Pressure']['Units']).to('hPa')
        except:
            print('not using Sea Level Pressure data (either it is not existent\
 or it was not specified in the fvars file')
        
def main():
    DataObj = DataObject(NetCDF_data,dfVars,dfbox)
    TimeName = dfVars.loc['Time']['Variable']
    hgt = DataObj.GeopotHeight
    omega = DataObj.Omega
    u,v = DataObj.u, DataObj.v
    temperature = DataObj.Temperature
    pres = DataObj.Pressure
    try:
        slp = DataObj.SLP
    except:
        slp = None
    for t in hgt[TimeName].values:
        # Plot hgt, omega and wind for some levels
        ight = hgt.sel({TimeName:t})
        ight.name = 'geo. height'
        iomega = omega.sel({TimeName:t})
        iomega.name = 'omega vel.'
        iu,iv = u.sel({TimeName:t}), v.sel({TimeName:t})
        plot_levels(iomega,outdir,'hgt_omega',ight,iu,iv)
        # plot theta, hgt and wind for 850 hPa
        itemperature = temperature.sel({TimeName:t})
        theta = potential_temperature(pres,itemperature)
        plot_ThetaHgtWind(theta,ight,iu,iv,outdir,'theta_hgt')
        # plot slp, jet and wind
        if slp is not None:
            islp = slp.sel({TimeName:t})
            plot_SLPJetWind(islp,iu,iv,outdir,'SLP_jet')
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "\
Program for synoptic analysis of an atmospheric system or region. \n \
A box is defined in the 'box_lims' file and then the analysis is performed \
within it. An auxilliary 'fvars' file is also needed: it contains the \
specified names used for each variable. The results are stored in the \
~'SynopticAnalysis_Results' directory")
    parser.add_argument("infile", help = "input .nc file with temperature,\
 geopotential and meridional, zonal and vertical components of the wind,\
  in pressure levels")
    parser.add_argument("-g", "--geopotential", default = False,
    action='store_true', help = "use the geopotential data instead of\
 geopotential height. The file fvars must be adjusted for doing so.")
    parser.add_argument("-b", "--box_limits", default = False,
    action='store_true', help = "use the box_limits file for slicing the data\
 for a desired domain.")
 
    args = parser.parse_args()
    # args = parser.parse_args(["-b","-g",
    #     "../data_etc/netCDF_files/Reg1-Yakecan_ERA5.nc"])
    
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    if args.box_limits:
        dfbox = pd.read_csv('./box_limits',header=None,delimiter=';',
                        index_col=0)
        print('using box_limits file for slicing data')
    else:
        dfbox = None
    
    infile  = args.infile
    NetCDF_data = convert_lon(xr.open_dataset(infile),
                              dfVars.loc['Longitude']['Variable'])
    
    output = infile.split('/')[-1].split('.')[0]
    outdir = "../SynopticAnalysis_Results/"+output+"/"; os.makedirs(
        outdir, exist_ok=True)
    
    # Run the program
    main()            