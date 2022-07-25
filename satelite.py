#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:03:17 2022

@author: danilocoutodsouza
"""

from netCDF4 import Dataset              # Read / Write NetCDF4 files
import matplotlib.pyplot as plt          # Plotting library
from datetime import datetime            # Basic Dates and time types
import cartopy, cartopy.crs as ccrs      # Plot maps
import cartopy   
import os                              # Miscellaneous operating system interfaces
from utilities import download_CMI       # Our own utilities
from utilities import geo2grid, convertExtent2GOESProjection      # Our own utilities
import cartopy.io.shapereader as shpreader # Import shapefiles

from matplotlib import cm            # Colormap handling utilities
import numpy as np                   # Scientific computing with Python
from utilities import loadCPT  

import pandas as pd


# Input and output directories
input = "Samples"; os.makedirs(input, exist_ok=True)
output = "Output"; os.makedirs(output, exist_ok=True)
# Get figure limits
dfbox = pd.read_csv('./box_limits',header=None,delimiter=';',index_col=0)
min_lon = float(dfbox.loc['min_lon'].values)
max_lon = float(dfbox.loc['max_lon'].values)
min_lat = float(dfbox.loc['min_lat'].values)
max_lat = float(dfbox.loc['max_lat'].values)
extent = [min_lon, min_lat, max_lon, max_lat]
# AMAZON repository information 
# https://noaa-goes16.s3.amazonaws.com/index.html
bucket_name = 'noaa-goes16'
product_name = 'ABI-L2-CMIPF'
# Dates
day = 4
month = '01'

BAND = 13 #13 IR TRADICIONAL
# for num in range(0, 22,1):
hourinit =10
hourfinal = 24
interevaloMinutos = 61
# hourfinal = hourinit-1

#############################  TESTE  ######################
teste = 0
if teste == 1:
    hourinit =17
    hourfinal = hourinit -1
    interevaloMinutos = 61
    
################################################################

numH = hourinit - 1
while numH < hourfinal+1:
    numH = numH +1
    hour = numH
    # print (hour,hourinit, hourfinal )
    if hour == 24:
        day = day + 1
        hourinit = 0
        hourfinal = 14
        numH =  0
        hour = 0

    hour = str("{0:02}".format(hour) )
    # print(day)
    for num in range(0, 60,interevaloMinutos):
         minuto = num
         minuto = str("{0:02}".format(minuto) )
         # day = str("{0:02}".format(day) )
         # print(hour,minuto) 
          # for band in range(16):
         for band in range(1): # PARA RODAR Só UMA BAND!!!
            yyyymmddhhmn = '2019'+month+str("{0:02}".format(day))+hour+minuto
            band = band+1
            print(yyyymmddhhmn)
            band = BAND # PARA RODAR Só UMA BAND!!!
            # Download the file
            file_name = download_CMI(str(yyyymmddhhmn), band, input)
    
            #-----------------------------------------------------------------------------------------------------------
            # Open the GOES-R image
            file = Dataset(f'{input}/{file_name}.nc')
                               
            # Convert lat/lon to grid-coordinates
            lly, llx = geo2grid(extent[1], extent[0], file)
            ury, urx = geo2grid(extent[3], extent[2], file)
                    
            # Get the pixel values
            data = file.variables['CMI'][ury:lly, llx:urx]       
            #-----------------------------------------------------------------------------------------------------------
            # Compute data-extent in GOES projection-coordinates
            img_extent = convertExtent2GOESProjection(extent)
            #-----------------------------------------------------------------------------------------------------------
            # Choose the plot size (width x height, in inches)
            plt.figure(figsize=(9,10))
            
            # Use the Geostationary projection in cartopy
            ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0))
            
              # Define the color scale based on the channel
            if band <= 6:
                colormap = "gray"   # Black to white for visible channels
                prodname = "Reflectance (%)"
                vmin2 = 0                                                       # Min. value
                vmax2 = 1 
                img = ax.imshow(data, vmin=vmin2, vmax=vmax2, origin='upper', extent=img_extent, cmap='gray') #CINZA

            else:
                # colormap = "gray_r" # White to black for IR channels
                data = data - 273.15
                prodname = "Brightness Temperatures (°C)"
                vmin2 = -85                                                       # IR BAND13
                vmax2 = 80 
                
                # print (data)                                             # Max. value
                
                # gray_cmap = cm.get_cmap('gray_r', 120)                            # Read the reversed 'gray' cmap
                # gray_cmap = gray_cmap(np.linspace(0, 1, 120))                     # Create the array
                # colors = ["#ffffff","#ffa0ff",  "#feff65", "#66CD00","#0806ff", "#3bcfff"]  # Custom colors
                # my_colors = cm.colors.ListedColormap(colors) 
                
                fff=33                     # Create a custom colormap
                # my_colors = my_colors(np.linspace(0, 1, fff))                      # Create the array
                # gray_cmap[:fff, :] = my_colors                                     # Join both cmaps arrays
                # my_cmap2 = cm.colors.ListedColormap(gray_cmap)
                # # Add a colorbar
                gray_cmap = cm.get_cmap('gray_r', 120)                            # Read the reversed 'gray' cmap
                gray_cmap = gray_cmap(np.linspace(0, 1, 120))                     # Create the array
                jet_cmap  = cm.get_cmap('jet_r', 40)                              # Read the reversed 'jet' cmap 
                jet_cmap  = jet_cmap(np.linspace(0, 1, fff))                       # Create the array
                gray_cmap[:fff, :] = jet_cmap                                      # Join both cmaps arrays
                my_cmap2 = cm.colors.ListedColormap(gray_cmap)                    # Create the custom colormap

            
                # Plot the image
                # img = ax.imshow(data,  vmin=vmin2, vmax=vmax2,origin='upper', extent=img_extent, cmap='gray_r') #CINZA
                img = ax.imshow(data, vmin=vmin2, vmax=vmax2, origin='upper', extent=img_extent,cmap=my_cmap2) #Realçado
                
            # ADICIONA O X NA CIDADE
            # pirassununga_lon, pirassununga_lat = -51.33, -30.12
            
                  
            # plt.plot([pirassununga_lon], [pirassununga_lat],
            # color='black', markersize=15, marker='X', markerfacecolor='w',
            #   transform=ccrs.Geodetic()             ) 

            # import haversine as hs
            # loc1=(-22.1,-50.3)
            # loc2=(-19.1,-50.050308)
            # dist = hs.haversine(loc1,loc2)
            # # print(dist,'ggggggggg')  
            # import matplotlib.patches as mpatches
            # import cartopy.geodesic as catG
            
            
            # import shapely
            # example: draw circle with 45 degree radius around the North pole
            # lon = -53
            # lat = -26
            
            # r = 4
            # radius_in_meters =180000
            
            # find map ranges (with 5 degree margin)                       
            # CS = ax.contour(data,levels = [-52], vmin=vmin2, vmax=vmax2,
            #       colors=('r'),linestyles=('-'),origin='upper',linewidths=(1),extent=img_extent)
            # ax.contour(data,levels = [-33], vmin=vmin2, vmax=vmax2,
            #       colors=('b'),linestyles=('-'),origin='upper',linewidths=(1),extent=img_extent)
             
            # circle_points = catG.Geodesic().circle(lon=lon, lat=lat, radius=radius_in_meters, endpoint=False)
            # geom = shapely.geometry.Polygon(circle_points)
            # ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='g', alpha=0.2 ,edgecolor='white', linewidth=1 ,zorder=300)
           
            
                     # # Add a shapefile
            # https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2019/Brasil/BR/br_unidades_da_federacao.zip
            ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False

            shapefile = list(shpreader.Reader('br_unidades_da_federacao/BR_UF_2019.shp').geometries())
            ax.add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='white',facecolor='none', linewidth=0.3)
            
            # # Add coastlines, borders and gridlines
    
            ax.coastlines(resolution='10m', color='white', linewidth=0.8)
            ax.add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
            ticks = np.arange(-80,81,10)
            # ticks = [-80,-70, -61,-50, -38,-23, 0, 20, 40, 60 ,80]
            # Add a colorbar
            cb= plt.colorbar(img, label='Brightness Temperature (°C)', extend='both', orientation='vertical', pad=0.018, fraction=0.038)
            loc    = ticks
            cb.set_ticks(loc)
            cb.set_ticklabels(ticks)
            # Extract the date
            date = (datetime.strptime(file.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ'))
            
            # Add a title        
            plt.title('GOES-16 Band '+ str(band) +' ', fontweight='bold', fontsize=10, loc='left')
            plt.title('Valid: ' + date.strftime('%Y-%m-%d %H:%M') + ' UTC', fontsize=10, loc='right')
            #-----------------------------------------------------------------------------------------------------------
            
            if teste != 1:
                ############################# Save the image
                # plt.savefig(f'{output}/CONTORNO {file_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.savefig(f'{output}/Realçado'+BAND+''+day+ ' '+hour+minuto+' .png', bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.savefig(f'{output}/BAND '+str(BAND)+' '+str(month)+str(day)+ ' '+hour+minuto+' .png', bbox_inches='tight', pad_inches=0, dpi=300)
                plt.savefig(f'{output}/VCAN_BAND '+str(BAND)+' '+str(month)+str(day)+ ' '+hour+minuto+' .png', bbox_inches='tight', pad_inches=0, dpi=300)


            # Show the image
            plt.show()