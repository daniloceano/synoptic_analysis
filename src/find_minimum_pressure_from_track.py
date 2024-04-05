import os
import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.colors as mcolors

box_size = 5  # Half the size of the box, to create a 5x5 box


def find_real_track(first_guess_df, data_ds):
    """
    Optimize the cyclone track using the 'msl' variable from the data dataset.
    For each timestep of first_guess, find in the data the minimum 'msl' in a 5ºx5º box centered on the lat and lon values.
    
    Parameters:
    - first_guess_df: DataFrame containing the initial cyclone track guesses.
    - data_ds: xarray Dataset containing 'msl' and other variables.
    
    Returns:
    - DataFrame with updated lat, lon values and the minimum 'msl' found within a 5ºx5º box,
      formatted to one decimal place.
    """
    optimized_track = first_guess_df.copy()[["Lat", "Lon"]]

    # Convert longitude from [-180, 180] to [0, 360]
    data_ds.coords["longitude"] = (data_ds.coords["longitude"] + 180) % 360 - 180
    data_ds = data_ds.sortby(data_ds["longitude"])
    
    for index, row in first_guess_df.iterrows():
        # Define the 5ºx5º box boundaries
        lat_min = row['Lat'] - box_size
        lat_max = row['Lat'] + box_size
        lon_min = row['Lon'] - box_size
        lon_max = row['Lon'] + box_size
        
        # Filter the data within the box for the corresponding time
        data_filtered = data_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max), time=index)
        
        # Find the minimum 'msl' value and its location
        min_msl_value = data_filtered['msl'].min()
        min_msl_location = np.where(data_filtered['msl'] == min_msl_value)

        # Round the minimum 'msl' value to one decimal place
        min_msl_value_rounded = round(float(min_msl_value), 1)/100
        
        # Update the lat, lon in optimized_track with the location of the minimum 'msl'
        optimized_lat = data_filtered.latitude.values[min_msl_location[0]][0]
        optimized_lon = data_filtered.longitude.values[min_msl_location[1]][0]
        
        optimized_track.at[index, 'Lat'] = optimized_lat
        optimized_track.at[index, 'Lon'] = optimized_lon
        optimized_track.at[index, 'Min_MSL'] = min_msl_value_rounded
    
    return optimized_track

def plot_tracks(original_track, optimized_track, figures_directory):
    """
    Plots the original and optimized tracks of a weather system and saves the figure.

    Parameters:
    - original_track (pandas.DataFrame): DataFrame containing the original track data.
    - optimized_track (pandas.DataFrame): DataFrame containing the optimized track data.
    - figures_directory (str): Directory path to save the plot.
    """
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    # Combine tracks for calculating extents
    combined_track = pd.concat([original_track, optimized_track])
    
    min_lon, max_lon = combined_track['Lon'].min(), combined_track['Lon'].max()
    min_lat, max_lat = combined_track['Lat'].min(), combined_track['Lat'].max()
    
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5], crs=ccrs.PlateCarree())
    
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor="lightblue")
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, color='darkgray')
    gl.top_labels = gl.right_labels = False
    
    # Plotting the tracks
    ax.plot(original_track['Lon'], original_track['Lat'], 'r-', label='Original Track')
    ax.plot(optimized_track['Lon'], optimized_track['Lat'], 'b--', label='Optimized Track')
    
    ax.legend(loc="best")
    
    filename = os.path.join(figures_directory, 'track_comparison.png')
    plt.savefig(filename, bbox_inches='tight')  
    plt.close()
    print(f"Track comparison saved to: {filename}")  

def plot_msl_and_tracks_for_timestep(original_track, optimized_track, data_ds, index, timestep, figures_directory):
    """
    Plots MSL, a 5ºx5º box, and positions of the original and optimized tracks for a given timestep,
    adjusting MSL values and using a specified colormap and normalization.

    Parameters:
    - original_track (pandas.DataFrame): DataFrame with the original track data.
    - optimized_track (pandas.DataFrame): DataFrame with the optimized track data.
    - data_ds (xarray.Dataset): Dataset containing 'msl' and other variables.
    - index: The index of the timestep in the DataFrame.
    - timestep: The specific timestep to plot.
    - figures_directory (str): Directory path to save the plot.
    """
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    orig_pos = original_track.iloc[index]
    opt_pos = optimized_track.iloc[index]

    # Define the 5ºx5º box boundaries
    lat_min = orig_pos['Lat'] - box_size
    lat_max = orig_pos['Lat'] + box_size
    lon_min = orig_pos['Lon'] - box_size
    lon_max = orig_pos['Lon'] + box_size

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor="lightblue")

    # Adjust MSL values by dividing by 100
    msl_data = data_ds['msl'].sel(time=timestep) / 100

    # Define the colormap and normalization
    levels = np.linspace(990, 1030, 20)
    # norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
    cmap = plt.get_cmap('bwr')

    # Plot MSL with specified colormap and normalization
    cf = plt.contourf(msl_data["longitude"], msl_data["latitude"], msl_data, levels=levels, cmap=cmap, extend='both')
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=16, shrink=0.8, extend='both')
    cbar.set_label('MSL (hPa)')

    # Draw the 5ºx5º box
    box = mpatches.Rectangle((lon_min, lat_min), box_size * 2, box_size * 2, 
                             transform=ccrs.PlateCarree(), 
                             fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(box)

    # Mark the original and optimized positions
    ax.plot(orig_pos['Lon'], orig_pos['Lat'], 'ro', label='Original Position')
    ax.plot(opt_pos['Lon'], opt_pos['Lat'], 'bx', label='Optimized Position')

    ax.legend(loc="best")

    filename = os.path.join(figures_directory, f'msl_track_{index}.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot to save memory

# Loading the data
infile = '../../Programs_and_scripts/data_etc/netCDF_files/akara_subset_slp.nc'
data = xr.open_dataset(infile)
first_guess = pd.read_csv('inputs/first_guess', sep=';', index_col=0, header=0)

# Optimizing the track
optimized_track = find_real_track(first_guess, data)

# Create the output directory
filename = os.path.basename(infile).split('.')[0]
results_directory = f"../synoptic_analysis_results/{filename}/" 
os.makedirs(results_directory, exist_ok=True)

# Save the optimized track to a CSV file
optimized_track_file_path = f"./{results_directory}/optimized_track.csv"
optimized_track.to_csv(optimized_track_file_path, sep=';', index=True)

# Plot the tracks
plot_tracks(first_guess, optimized_track, results_directory)

# Plot the MSL and tracks for each timestep
for index, (timestamp, _) in enumerate(first_guess.iterrows()):
    print(f"Plotting MSL and tracks for timestep {index}")
    plot_msl_and_tracks_for_timestep(first_guess, optimized_track, data, index, timestamp, results_directory)