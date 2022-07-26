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
from metpy.constants import Cp_d
from metpy.constants import Re
from metpy.calc import potential_temperature

import pandas as pd
import xarray as xr
import os
import numpy as np
import argparse

from calc import CalcAreaAverage, CalcZonalAverage
from plot_maps import LagrangianMaps as plot_map

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
        self.GeopotHeight = self.NetCDF_data[dfVars.loc['Geopotential Height']['Variable']] \
            * units(dfVars.loc['Geopotential Height']['Units']).to('gpm')
        self.Pressure = self.NetCDF_data[self.LevelIndexer]
        
def main():
    DataObj = DataObject(NetCDF_data,dfVars)
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "\
Program for synoptic analysis of an atmospheric system or region. \n \
A box is definid in the box_lims' file and then the analysis is performed \
within it. An auxilliary 'fvars' file is also needed for both frameworks: it \
contains the specified names used for each variable. The results are stored \
in the 'SynopticAnalysis_Results' directory")
    parser.add_argument("infile", help = "input .nc file with temperature,\
  geopotential and meridional, zonal and vertical components of the wind,\
  in pressure levels")
    parser.add_argument("-g", "--geopotential", default = False,
    action='store_true', help = "use the geopotential data instead of\
  geopotential height. The file fvars must be adjusted for doing so.")

    args = parser.parse_args()
    
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    dfbox = pd.read_csv('./box_limits',header=None,delimiter=';',
                        index_col=0)
    
    infile  = args.infile
    NetCDF_data = convert_lon(xr.open_dataset(infile),
                              dfVars.loc['Longitude']['Variable'])
    
    # Run the program
                      