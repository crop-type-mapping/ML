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
import time
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


import ee
import geemap
from datetime import datetime

# Initialize Earth Engine
geemap.ee_initialize()
print('GeeMap initialized!')

# ============================================
# Parameters
# ============================================
eyear = 2025
start = ee.Date(f'{eyear}-02-01')
end = ee.Date(f'{eyear}-06-30')
timeField = 'system:time_start'
sResolution = 10
season = 'B'
version = 'v1'
filename = f'S1_Multitemp_{eyear}_{season}_{version}'
gee_project = "projects/cropmapping-365811"
asset_folder = "rwanda"
s1_asset_id = f"{gee_project}/assets/{asset_folder}/{filename}"
# ============================================
# Load ROI and display
# ============================================
cList = ['Nyagatare', 'Musanze', 'Ruhango', 'Nyabihu']
ROI = ee.FeatureCollection('users/bensonkemboi/CIAT/Rwanda/rwa_adm2_selected_districts')
ROI = ROI.filter(ee.Filter.inList('ADM2_EN', cList))
Map = geemap.Map(center=[-1.94, 29.87], zoom=8)
Map.addLayer(ROI, {}, 'ROI')
# ============================================
# Load Mask 
# ============================================
# Load ESA WorldCover (v200 - 2020 or latest available)
esa_landcover = ee.ImageCollection("ESA/WorldCover/v200").first()
# Select the 'Map' band and create mask for class values: 90 (Cropland), 40 (Wetland)
cropland = esa_landcover.select('Map').eq(90)
wetland = esa_landcover.select('Map').eq(40)
# Combine and apply mask
datamask = cropland.Or(wetland).selfMask().rename('cropland')

# ============================================
# Sentinel-1 images
# ============================================
s1Base = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(ROI) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .filter(ee.Filter.notNull(['system:time_start']))
# ============================================
# CHIRPS & ERA5
# ============================================
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(ROI)
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end).filterBounds(ROI)

# ============================================
# Helper functions
# ============================================
def toNatural(img): return ee.Image(10.0).pow(img.divide(10.0))
def toDB(img): return ee.Image(img).log10().multiply(10.0)
def maskBorderNoise(img): return img.updateMask(img.select('VV').mask())

def refinedLee(img):
    img = toNatural(img)
    kernel = ee.Kernel.fixed(3, 3, [[1]*3]*3, 1, 1, False)
    mean = img.reduceNeighborhood(ee.Reducer.mean(), kernel)
    variance = img.reduceNeighborhood(ee.Reducer.variance(), kernel)
    enl = 4.4
    noiseVar = mean.pow(2).divide(enl)
    b = variance.divide(variance.add(noiseVar))
    result = mean.add(b.multiply(img.subtract(mean)))
    return toDB(result.rename(img.bandNames()))

def toGamma0(img):
    return img.expression(
        'i - 10 * log10(cos(angle * pi / 180))',
        {'i': img.select(['VV', 'VH']), 'angle': img.select('angle'), 'pi': 3.1415927}
    ).rename(['VV', 'VH']).copyProperties(img, img.propertyNames())

def addRadarIndices(img):
    vv = img.select('VV')
    vh = img.select('VH')
    rvi = img.expression('(4 * VH) / (VV + VH)', {'VV': vv, 'VH': vh})
    cri = img.expression('VV / VH', {'VV': vv, 'VH': vh})
    mrvi = img.expression('((VV / (VV + VH)) ** 0.5) * ((4 * VH) / (VV + VH))', {'VV': vv, 'VH': vh})
    mrfdi = img.expression('(VV - VH) / (VV + VH)', {'VV': vv, 'VH': vh})
    indices = ee.Image.cat([cri, mrfdi, mrvi, rvi]).rename(['cri', 'mrfdi', 'mrvi', 'rvi'])
    return img.addBands(indices.float())

def addVariables(img):
    years = ee.Date(img.get('system:time_start')).difference(ee.Date('2017-01-01'), 'year')
    return img.addBands(ee.Image(years).rename('t').float()) \
              .addBands(ee.Image.constant(1)) \
              .updateMask(datamask)
# ============================================
# Generate Monthly Composites
# ============================================
def get_monthly_composites(start, end):
    months = ee.List.sequence(0, end.difference(start, 'month').subtract(1))
    def monthly_fn(m):
        m = ee.Number(m)
        month_start = start.advance(m, 'month')
        month_end = month_start.advance(1, 'month')
        s1_monthly = s1Base.filterDate(month_start, month_end) \
            .map(addVariables).map(maskBorderNoise).map(lambda img: img.updateMask(datamask)) \
            .map(refinedLee).map(toGamma0).map(addRadarIndices)
        s1_comp = s1_monthly.median()
        s1_proj = s1_comp.select('VV').projection()
        '''
        rain = chirps.filterDate(month_start, month_end).select('precipitation').sum() \
            .reproject(crs=s1_proj.crs(), scale=sResolution).resample('bilinear').rename('ch_rain')
        evap = era5.select('total_evaporation_sum').mean() \
            .reproject(crs=s1_proj.crs(), scale=1000).resample('bilinear').rename('era_evap')
        temp = era5.select('temperature_2m').mean().subtract(273.15) \
            .reproject(crs=s1_proj.crs(), scale=1000).resample('bilinear').rename('era_t2m')
        '''

        combined = s1_comp.select(['VV', 'VH', 'cri', 'mrfdi', 'rvi']) \
            .set({
                'month': month_start.get('month'),
                'year': month_start.get('year'),
                'month_id': month_start.format('YYYY_MM'),
                'system:time_start': month_start.millis()
            })

        s1_obia = ee.Algorithms.Image.Segmentation.SNIC(
            combined.clip(ROI),  # image
            30,                  # size
            0.1,                 # compactness
            8,                   # connectivity
        ).select('clusters').rename('s1_obia')

        return combined.addBands(s1_obia).updateMask(datamask)

    return months.map(monthly_fn)

# Create monthly ImageCollection
monthlyComposites = ee.ImageCollection.fromImages(get_monthly_composites(start, end))
first_image = monthlyComposites.first()
band_names = first_image.bandNames().getInfo()
print('Monthly composites (S1):', band_names)


# Visualize first month
Map = geemap.Map()
Map.centerObject(ROI, 8)
Map.addLayer(monthlyComposites.first().clip(ROI),
             {'bands': ['VV', 'VH', 'rvi'], 'min': [-0.25, -0.25, -1], 'max': [5, 5, 1]},
             'First Monthly Composite')
Map
# ============================================
# Stack images by renaming bands with month index
# ============================================
image_list = monthlyComposites.toList(monthlyComposites.size())
n_months = monthlyComposites.size().getInfo()

# Initialize band_stack with the first image (renamed)
img0 = ee.Image(image_list.get(0))
band_names0 = img0.bandNames()
new_names0 = band_names0.map(lambda b: ee.String(b).cat('_1'))
band_stack = img0.rename(new_names0)

# Loop through the rest starting from index 1
for i in range(1, n_months):
    index = i + 1
    img = ee.Image(image_list.get(i))
    band_names = img.bandNames()
    new_names = band_names.map(lambda b: ee.String(b).cat('_').cat(ee.Number(index).format()))
    renamed = img.rename(new_names)
    band_stack = band_stack.addBands(renamed)

# ============================================
# Export the stacked image
# ============================================
# Convert ImageCollection to list
imageList = monthlyComposites.toList(monthlyComposites.size())
nMonths = monthlyComposites.size().getInfo()

# Initialize bandStack with the first image (after renaming)
first_img = ee.Image(imageList.get(0))
first_bands = first_img.bandNames().map(lambda b: ee.String(b).cat('_1'))
bandStack = first_img.rename(first_bands)

# Loop over remaining images (start from index 1)
for i in range(1, nMonths):
    index = i + 1  # For naming bands starting from 1
    img = ee.Image(imageList.get(i))
    bandNames = img.bandNames()
    renamed = img.rename(bandNames.map(lambda b: ee.String(b).cat(f'_{index}')))
    bandStack = bandStack.addBands(renamed)


# Define the Asset ID path
s1_asset_id = f"{gee_project}/assets/{asset_folder}/{filename}"

# Delete existing asset if exists (overwrite support)
try:
    ee.data.deleteAsset(s1_asset_id)
    print("ℹ️ Existing asset deleted to allow overwrite.")
except Exception as e:
    print("✅ No existing asset to delete (safe to proceed).")

# Define Export Task
export_task = ee.batch.Export.image.toAsset(
    image=bandStack.clip(ROI),
    description=filename,
    assetId=s1_asset_id,
    region=ROI.geometry(),
    scale=sResolution,
    crs='EPSG:4326',
    maxPixels=1e13
)

# Start Task
export_task.start()
print(f"Export task '{filename}' started...")

# Monitor Task Status
while export_task.active():
    print("Export in progress... waiting 30s...")
    time.sleep(30)

# Final status
status = export_task.status()
print(f"Export task completed with status: {status['state']}")

# Handle potential failure
if status['state'] == 'FAILED':
    print(f"Error message: {status.get('error_message', 'No error message provided.')}")


print(f"Pre-processed raster saved to {s1_asset_id}")
print("Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)