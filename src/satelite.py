#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:09:03 2022

Created by:
    Danilo Couto de Souza
    Universidade de São Paulo (USP)
    Instituto de Astornomia, Ciências Atmosféricas e Geociências
    São Paulo - Brazil
    
Based on scripts available by the Brazilian National Space Research Institute
(INPE), available at: https://moodle.cptec.inpe.br/course/view.php?id=10

Contact:
    danilo.oceano@gmail.com
"""
# Training: Python and GOES-R Imagery: Script 9 - Downloading data from AWS
#-----------------------------------------------------------------------------------------------------------
# Required modules
from netCDF4 import Dataset          # Read / Write NetCDF4 files
import matplotlib.pyplot as plt      # Plotting library
import datetime                      # Basic Dates and time types
import cartopy, cartopy.crs as ccrs  # Plot maps
import os                            # Miscellaneous operating system interfaces
import boto3                         # Amazon Web Services (AWS) SDK for Python
from botocore import UNSIGNED        # boto3 config
from botocore.config import Config   # boto3 config
from utilities import geo2grid, convertExtent2GOESProjection      # Our own utilities
from matplotlib import cm            # Colormap handling utilities
import numpy as np                   # Scientific computing with Python
from utilities import loadCPT        # Import the CPT convert function
import sys
import pandas as pd
import xarray as xr
import argparse
#-----------------------------------------------------------------------------------------------------------
# Function to download files
def download_file(s3_client, prefix):
    # Seach for the file on the server
    s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter = "/")
    # Check if there are files available
    if 'Contents' not in s3_result:
        # There are no files
        print("No files found for the date: ",year,day_of_year)
        sys.exit()
    else:
        # There are files
        for obj in s3_result['Contents']:
            # Print the file name
            key = obj['Key']
            print(key)
            file_name = key.split('/')[-1].split('.')[0]
      
            # Download the file
            if not os.path.exists(f'{indir}/{file_name}.nc'):
                s3_client.download_file(bucket_name, key, f'{indir}/{file_name}.nc')

    return file_name
#-----------------------------------------------------------------------------------------------------------
def make_img(year,day_of_year,hour,minute,band):
    # Initializes the S3 client
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    #-----------------------------------------------------------------------------------------------------------
    # File structure
    prefix = f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{band:02.0f}_G16_s{year}{day_of_year:03.0f}{hour:02.0f}{minute:02.0f}'
    # Download the file
    file_name = download_file(s3_client, prefix)
    #-----------------------------------------------------------------------------------------------------------
    # Open the GOES-R image
    file = Dataset(f'{indir}/{file_name}.nc')  
    
    # Convert lat/lon to grid-coordinates
    lly, llx = geo2grid(extent[1], extent[0], file)
    ury, urx = geo2grid(extent[3], extent[2], file)      
    
    # Get the pixel values
    data = file.variables['CMI'][:][ury:lly, llx:urx]
    #-----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.close('all')
    plt.figure(figsize=(10,10))
    
    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.Geostationary(
        central_longitude=min_lon-10,
        satellite_height=35786023.0))
    
    # Compute data-extent in GOES projection-coordinates
    img_extent = convertExtent2GOESProjection(extent)
        
    # Define the color scale based on the channel
    if band <= 6:
        prodname = "Reflectance (%)"
        vmin2 = 0                                                       # Min. value
        vmax2 = 1 
        # Plot the image
        img = ax.imshow(data, vmin=vmin2, vmax=vmax2, origin='upper',
                        extent=img_extent, cmap='gray')
    else:
        data = data - 273.15
        prodname = "Brightness Temperatures (°C)"
        prodname = "Brightness Temperatures (°C)"
        vmin2 = -85                                                       # IR BAND13
        vmax2 = 80 
      
        fff=33
        gray_cmap = cm.get_cmap('gray_r', 120)                            # Read the reversed 'gray' cmap
        gray_cmap = gray_cmap(np.linspace(0, 1, 120))                     # Create the array
        jet_cmap  = cm.get_cmap('jet_r', 40)                              # Read the reversed 'jet' cmap 
        jet_cmap  = jet_cmap(np.linspace(0, 1, fff))                       # Create the array
        gray_cmap[:fff, :] = jet_cmap                                      # Join both cmaps arrays
        my_cmap2 = cm.colors.ListedColormap(gray_cmap)                    # Create the custom colormap
        # Plot the image
        img = ax.imshow(data, vmin=vmin2, vmax=vmax2, origin='upper',
                        extent=img_extent,cmap=my_cmap2) #Realçado  
    
    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='white', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='white',
                    linewidth=0.5)
    ax.gridlines(color='white', alpha=0.5, linestyle='--', 
                  linewidth=0.5) 
    
    
    # Add a colorbar
    plt.colorbar(img, label=prodname, extend='both', 
                 orientation='horizontal', pad=0.05, fraction=0.05)
    
    # Extract date
    date = (datetime.datetime.strptime(file.time_coverage_start,
                                       '%Y-%m-%dT%H:%M:%S.%fZ'))
    
    # Add a title
    plt.title('GOES-16 Band-' + str(band) + ' ' + date.strftime(
        '%Y-%m-%d %H:%M') + ' UTC', fontweight='bold', fontsize=10,
        loc='left')
    plt.title('Full Disk', fontsize=10, loc='right')
    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    prefix_name = bucket_name+'_band'+str(band)+'_'+product_name+'_'
    fname = f'{output}/'+prefix_name+date.strftime('%Y%m%d%H%M%S')+'.png'
    plt.savefig(fname)
    print(fname+" created")
#-----------------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "\
Program for generating images from the GOES16, Band 13 for a system speficied \
An auxilliary 'fvars' file is also needed for both frameworks: it contains \
the specified names used for each variable.  The results are stored in the \
'SynopticAnalysis_Results/Satelite' directory")
    parser.add_argument("infile", help = "input .nc file")
    
    # Input and output directoriesindir = "./Samples"; os.makedirs(indir, exist_ok=True)
    indir = "Samples"; os.makedirs(indir, exist_ok=True)
    infile = parser.parse_args().infile
    outdir = infile.split('/')[-1].split('.')[0]
    output = "../SynopticAnalysis_Results/"+outdir+"/Satelite/"; os.makedirs(
        output, exist_ok=True)
    
    # Get figure limits
    dfbox = pd.read_csv('./box_limits',header=None,delimiter=';',
                        index_col=0)
    min_lon = float(dfbox.loc['min_lon'].values)
    max_lon = float(dfbox.loc['max_lon'].values)
    min_lat = float(dfbox.loc['min_lat'].values)
    max_lat = float(dfbox.loc['max_lat'].values)
    extent = [min_lon, min_lat, max_lon, max_lat]
    
    # Get time coverage from file
    dfVars = pd.read_csv('./fvars',sep= ';',index_col=0,header=0)
    ncfile = xr.open_dataset(infile)
    nctime = ncfile[dfVars.loc['Time']['Variable']]


    # AMAZON repository information 
    # https://noaa-goes16.s3.amazonaws.com/index.html
    bucket_name = 'noaa-goes16'
    product_name = 'ABI-L2-CMIPF'
    band = 13
    for timestep in nctime:
        ts = datetime.datetime.strptime(str(timestep.values
                                            )[:16],'%Y-%m-%dT%H:%M')
        year = ts.year
        day_of_year = ts.timetuple().tm_yday
        hour = ts.hour
        minute = ts.minute
        make_img(year,day_of_year,hour,minute,band)