#!/usr/bin/env python
# coding: utf-8

# Code reference: <a href="https://githubtocolab.com/giswqs/geemap/blob/master/examples/notebooks/46_local_rf_training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# ## Set up Librareis and variables

# Export Images from GEE using: https://code.earthengine.google.com/74ceb413aa5dfdfe044e22f0b128625d
# 
# Before starting, assumming you are working in Linux, server do the following:
# 1. Create a virtual environment e.g. `conda create -n my-env` where `my-env` is the name of the virtual environment,
# 2. Activate the environment and install Jupyter notebooks `pip3 install --user ipykernel` and check if installed by running `which jupyter`,
# 3. Associate the notebook with your environment `python -m ipykernel install --user --name=crops-env`.
# 4. Install necessary packages (amongs others), e.g.`conda install -c conda-forge google-colab`, `conda install geemap -c conda-forge`, `conda install -c conda-forge matplotlib`, `conda install -c conda-forge earthengine-api` and `conda install --channel conda-forge geopandas`. Then set-up a service account for GEE cloud service account.
import os
import ee
import json
from google.auth import credentials
from google.oauth2 import service_account
import timeit
start_time = timeit.default_timer() 
# Path to your service account key JSON file
SERVICE_ACCOUNT_KEY = "gee_key.json"
 
# Load credentials from the service account file
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_KEY,
    scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/drive']  # Add Drive scope if needed
)
 
# Initialize Earth Engine API
ee.Initialize(credentials)
 
# Verify successful initialization
print("Earth Engine initialized successfully!")
# ## Get Libraries

#!pip install geemap scikit-learn
import geemap
import pandas as pd
from geemap import ml
from sklearn import ensemble
import time

# Initialize GEE

geemap.ee_initialize()
print('GeeMap initialized!')

print("Libraries initialized successfully!")


# ### Define Global variables

gee_project = "projects/cropmapping-365811"  # Change to your GEE project name
asset_folder = "rwanda"
rawpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/RAW/'
interimpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/Process/'
finalpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/images/'
sResolution = 10 
sensor = "S1" #S1_Comp_Nyagatare_2019_B_v1
season ='B'
eyear  = '2025';
version = 'v1'

cList = ['Nyagatare']
roi = ee.FeatureCollection('users/bensonkemboi/CIAT/Rwanda/rwa_adm2_selected_districts')
roi = roi.filter(ee.Filter.inList('ADM2_EN', cList))

filename = f"{sensor}_Comp_Nyagatare_{eyear}_{season}_{version}"#'s1_composite_'+eyear+'_'+season+'_'+{version} 
temp   = f"{filename}_image_chunk"
output_merged_file = os.path.join(finalpath, f"{filename}_mosaic.tif")
tile_width = 0.08#0.065  
tile_height =  0.08 #0.065
no_datan = -9999
print("Global variables initialized succefully!")

# Here we will load training points from shapefile including AOI
s1_composite_asset_id = f"{gee_project}/assets/{asset_folder}/{filename}"

s1_composite = ee.Image(s1_composite_asset_id) # load from asssets
print(s1_composite.bandNames().getInfo())
print('Successfully loaded ', s1_composite_asset_id)

def create_grid(roi, tile_width, tile_height):
    """
    Create a list of rectangular tiles (ee.Geometry.Rectangle) covering the ROI.
    tile_width and tile_height are in the same units as ROI coordinates (e.g., degrees for EPSG:4326).
    """
    bounds = roi.bounds().getInfo()['coordinates'][0]
    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    minX, maxX = min(xs), max(xs)
    minY, maxY = min(ys), max(ys)
   
    grid = []
    x = minX
    while x < maxX:
        y = minY
        while y < maxY:
            tile = ee.Geometry.Rectangle([x, y, min(x + tile_width, maxX), min(y + tile_height, maxY)])
            grid.append(tile)
            y += tile_height
        x += tile_width
    return grid
 

 
tiles = create_grid(roi, tile_width, tile_height)
print(f"Created {len(tiles)} tiles.")

# ### Export tiles to file

from osgeo import gdal
import os
new_image = s1_composite.unmask(no_datan)
band_names = new_image.bandNames().getInfo()
print('Bands ', band_names)
# Loop through tiles and download
for i, tile in enumerate(tiles):
    # Define output filename imgname
    imgname = f"{temp}*.tif"  #output chunk name
    #imgname = f"{sensor}_{eyear}_{season}_image_chunk_v1*.tif" #output chunk name
    filename = f"{temp}_{i}.tif"#filename = f"{sensor}_{eyear}_{season}_image_chunk_v2_{i}.tif"
    filepath = os.path.join(interimpath, filename)
    
    # ✅ Skip if file already exists
    if os.path.exists(filepath):
        print(f"Tile {i} already exists. Skipping...")
        continue
    
    print(f"Exporting Tile {i} to {filepath}...")

    try:
        # Export using geemap (downloads directly)
        geemap.ee_export_image(
            new_image.clip(tile),
            filepath,
            scale=sResolution,
            region=tile,
            file_per_band=False  # Saves as a single multiband GeoTIFF
        )
        print(f"Tile {i} saved to {filepath}")

        # ✅ Add band names using GDAL
        print(f"Updating band names for {filepath}...")
        dataset = gdal.Open(filepath, gdal.GA_Update)
        if dataset:
            # ✅ Set band names individually
            for band_index, band_name in enumerate(band_names, start=1):
                band = dataset.GetRasterBand(band_index)
                band.SetDescription(band_name)  # Set band description (for internal GDAL use)
                band.SetNoDataValue(no_datan)  # ✅ Set nodata here

            # ✅ Set metadata at the file level
            metadata = {f"Band_{i + 1}": band_name for i, band_name in enumerate(band_names)}
            dataset.SetMetadata(metadata)

            dataset.FlushCache()  # ✅ Ensure changes are written to disk
            dataset = None  # ✅ Close and save changes
            print(f"Band names added to {filepath}")

    except Exception as e:
        print(f"Error exporting tile {i}: {e}")

print('✅ Completed saving tiles')

# ✅ Optional post-processing: convert -9999 to np.nan after loading with rasterio or gdal
# import rasterio
# with rasterio.open(filepath) as src:
#     data = src.read(1)
#     data[data == -9999] = np.nan

import glob
print('Number of created tiles...', len(tiles))
# Define input and output file paths


input_tiles_path = os.path.join(interimpath, imgname)

# ✅ Get list of tile files (filtering invalid filenames)
tile_files = glob.glob(input_tiles_path)
tile_files = [f for f in tile_files if '(' not in f and ')' not in f]  

print("Found tile files:", len(tile_files))

# ✅ Check if the number of downloaded tiles matches expected tiles
if len(tiles) != len(tile_files):
    print(f"Number of tiles downloaded ({len(tiles)}) does not match number of saved files ({len(tile_files)})!")
else:
    # ✅ Remove existing output file if present
    if os.path.exists(output_merged_file):
        os.remove(output_merged_file)

    print(f"Mosaicking {len(tile_files)} tiles into {output_merged_file}...")

    if len(tile_files) > 0:
        try:
            # ✅ Use subprocess to merge tiles
            #gdal_merge_command = ["gdal_merge.py", "-o", output_merged_file, "-of", "GTiff"] + tile_files
            # ✅ Use subprocess to merge tiles with compression
            no_datan =-9999 
            gdal_merge_command = [
                "gdal_merge.py",
                "-o", output_merged_file,
                "-of", "GTiff",
                "-co", "COMPRESS=DEFLATE",        # 🧵 Add compression
                "-co", "TILED=YES",               # 🧱 Improves performance for large rasters
                "-a_nodata", str(no_datan),
                "-co", "BIGTIFF=IF_SAFER"         # 🛡️ Enables BigTIFF if needed
            ] + tile_files
            os.system(" ".join(gdal_merge_command))
            print(f"Merged image saved to {output_merged_file}")

            # ✅ Preserve band names from the first input tile
            print("Preserving band names...")
            first_tile = tile_files[0]
            input_ds = gdal.Open(first_tile)
            output_ds = gdal.Open(output_merged_file, gdal.GA_Update)

            if input_ds and output_ds:
                num_bands = input_ds.RasterCount
                for band_index in range(1, num_bands + 1):
                    input_band = input_ds.GetRasterBand(band_index)
                    output_band = output_ds.GetRasterBand(band_index)

                    # ✅ Copy band description (band name)
                    band_name = input_band.GetDescription()
                    if band_name:
                        output_band.SetDescription(band_name)

                # ✅ Set metadata at file level
                band_names = [input_ds.GetRasterBand(i + 1).GetDescription() for i in range(num_bands)]
                metadata = {f"Band_{i + 1}": name for i, name in enumerate(band_names) if name}
                output_ds.SetMetadata(metadata)

                output_ds.FlushCache()
                output_ds = None
                input_ds = None

                print("✅ Band names preserved in the output file.")

        except Exception as e:
            print(f"Error during merging: {e}")
    else:
        print("No tile files found for merging!")
print(f"✅ Masaicked raster saved to {output_merged_file}")
print("Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)