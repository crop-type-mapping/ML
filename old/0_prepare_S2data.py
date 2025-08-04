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
import ee
import geemap
from datetime import datetime
import math

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
cloud_threshold = 0.50
filename = f'S2_Multitemp_{eyear}_{season}_{version}'
gee_project = "projects/cropmapping-365811"
asset_folder = "rwanda"
s2_asset_id = f"{gee_project}/assets/{asset_folder}/{filename}"
bands = ['B2', 'B3', 'B4', 'B7', 'B8', 'B11', 'B12'];
new_names = ['blue', 'green', 'red', 're3', 'nir', 'swir1', 'swir2'];
# ============================================
# Load ROI and display
# ============================================
cList = ['Musanze']#['Nyagatare', 'Musanze', 'Ruhango', 'Nyabihu']
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
cropland = esa_landcover.select('Map').eq(90) #Vegetattive Wetlands
wetland = esa_landcover.select('Map').eq(40) #Croplands
# Combine and apply mask
datamask = cropland.Or(wetland).selfMask().rename('cropland')

# ============================================
# Spectral Indices and helper functions
# ============================================
index_list = ['cvi', 'gvmi', 'ndvi', 'ndmi']
def add_indices(image):
    bands = {
        'blue': image.select('blue'),
        'green': image.select('green'),
        'red': image.select('red'),
        'nir': image.select('nir'),
        'swir1': image.select('swir1'),
        'swir2': image.select('swir2'),
        're3': image.select('re3'),
    }
    cvi = image.expression('(nir/green) * (red + green)', bands)
    gvmi = image.expression('((re3+0.1)-(swir2+0.02)) / ((re3+0.1)+(swir2+0.02))', bands)
    ndvi = image.expression('(nir - red) / (nir + red)', bands)
    ndmi = image.expression('(nir - swir1) / (nir + swir1)', bands)
    indices = ee.Image.cat([cvi, gvmi, ndvi, ndmi]).rename(['cvi', 'gvmi', 'ndvi', 'ndmi'])
    return image.addBands(indices.float())
#  Debug helper to print safely
def _dbg(msg, value):
    try:
        print(msg, value.getInfo())
    except Exception as e:
        print(msg, f"(getInfo failed: {e})")
'''
def add_time(image):
    date = ee.Date(image.get('system:time_start'))
    days = date.difference(ee.Date.fromYMD(eyear, 1, 1), 'day')
    radians = days.multiply(2 * 3.1416 / 365.25)
    return image.addBands([
        ee.Image.constant(days).rename('t').toFloat(),
        ee.Image.constant(radians.sin()).rename('sin').toFloat(),
        ee.Image.constant(radians.cos()).rename('cos').toFloat()
    ])
'''
def add_time(image):
    date = ee.Date(image.get('system:time_start'))
    days = date.difference(ee.Date.fromYMD(eyear, 1, 1), 'day')  # days since Jan 1 of eyear
    radians = days.multiply(2 * 3.1416 / 365.25)
    return image.addBands([
        ee.Image.constant(days).rename('t').toFloat(),
        ee.Image.constant(radians.sin()).rename('sin').toFloat(),
        ee.Image.constant(radians.cos()).rename('cos').toFloat()
    ], overwrite=True)
    

def monthly_composites(collection, start, end):
    """Create monthly median composites."""
    n_months = end.difference(start, 'month').round().toInt()
    months = ee.List.sequence(0, n_months.subtract(1))

    def make_month(m):
        m = ee.Number(m)
        start_m = start.advance(m, 'month')
        end_m = start_m.advance(1, 'month')
        monthly = collection.filterDate(start_m, end_m).median()
        return monthly.set('system:time_start', start_m.millis())

    return ee.ImageCollection(months.map(make_month))

# ============================================
# Harmonic Decomposition
# ============================================
'''
def harmonic_decompose(collection, index_list, start, end, roi):
    """Performs harmonic decomposition and monthly prediction for each index."""
    collection = collection.map(add_time)
    monthly = monthly_composites(collection, start, end).map(add_time)

    def fit_index(index):
        # Add band presence tag
        def tag_has_band(img):
            return img.set('has_band', img.bandNames().contains(index))

        filtered = collection.map(tag_has_band).filter(ee.Filter.eq('has_band', True))

        # Check if sufficient images
        if filtered.size().getInfo() < 3:
            print(f"⚠️ Not enough valid images for index: {index}")
            return ee.Image.constant(0).rename(f"{index}_empty")

        # Harmonic regression
        reg = filtered.select(['t', 'sin', 'cos', index]) \
                      .reduce(ee.Reducer.linearRegression(3, 1))
        coeffs = reg.select('coefficients')
        a0 = coeffs.arrayGet([0, 0])
        a1 = coeffs.arrayGet([0, 1])
        a2 = coeffs.arrayGet([0, 2])

        amplitude = a1.pow(2).add(a2.pow(2)).sqrt().rename(f"{index}_amplitude")
        phase = a1.atan2(a2).rename(f"{index}_phase")

        # Predict monthly values without using .toBands()
        months = monthly.toList(monthly.size())
        num_months = monthly.size().getInfo()

        predicted_images = []
        for i in range(num_months):
            img = ee.Image(months.get(i))
            sin = img.select('sin')
            cos = img.select('cos')
            pred = a0.add(a1.multiply(sin)).add(a2.multiply(cos))
            band_name = f"{index}_m{str(i + 1).zfill(2)}"
            pred = pred.rename(band_name)
            predicted_images.append(pred)

        fitted_image = ee.Image.cat(predicted_images)

        snic = ee.Algorithms.Image.Segmentation.SNIC(
            fitted_image.clip(roi),
            compactness=30,
            connectivity=8,
            neighborhoodSize=128
        ).select('clusters').rename(f"{index}_snic")

        return fitted_image.addBands(amplitude).addBands(phase).addBands(snic)

    # Combine results from all indices
    result_images = [fit_index(index) for index in index_list]
    return ee.Image.cat(result_images)
'''
import re

# Harmonic Decomposition
 
import re
 
#  Debug helper to print safely
def _dbg(msg, value):
    try:
        print(msg, value.getInfo())
    except Exception as e:
        print(msg, f"(getInfo failed: {e})")
 
# Make sure add_time overwrites time bands if they already exist
def add_time(image):
    date = ee.Date(image.get('system:time_start'))
    days = date.difference(ee.Date.fromYMD(eyear, 1, 1), 'day')  # days since Jan 1 of eyear
    radians = days.multiply(2 * 3.1416 / 365.25)
    return image.addBands([
        ee.Image.constant(days).rename('t').toFloat(),
        ee.Image.constant(radians.sin()).rename('sin').toFloat(),
        ee.Image.constant(radians.cos()).rename('cos').toFloat()
    ], overwrite=True)
 
def harmonic_decompose(collection_in, index_list, start, end, roi):
    """
    Performs harmonic decomposition and monthly prediction for each index.
    """
 
    # -------------------
    # DEBUG: inputs
    # -------------------
    _dbg("[DBG] Input collection size:", collection_in.size())
    _dbg("[DBG] Input first image band names:", collection_in.first().bandNames())
 
    # 1) Time features ONLY for fitting (keep original collection for monthly)
    collection_fit = collection_in.map(add_time)
    _dbg("[DBG] Fit collection band names (first image):", collection_fit.first().bandNames())
 
    # Build monthly composites from the ORIGINAL (no time bands carried into median),
    #    then add time once to monthly images
    def _strip_time(img):
        return img.select(img.bandNames().removeAll(['t', 'sin', 'cos']))
    base_for_monthly = collection_in.map(_strip_time)
 
    monthly = monthly_composites(base_for_monthly, start, end).map(add_time)
 
    # -------------------
    # DEBUG: monthly
    # -------------------
    _dbg("[DBG] Monthly image count:", monthly.size())
    _dbg("[DBG] Monthly first image band names:", monthly.first().bandNames())
    _dbg("[DBG] Monthly first image has sin?:", monthly.first().bandNames().contains('sin'))
    _dbg("[DBG] Monthly first image has cos?:", monthly.first().bandNames().contains('cos'))
 
    def fit_index(index):
        # Clean index name to start with a letter (export-safe)
        index_clean = re.sub(r'^[^A-Za-z]+', '', index)
 
        # -------------------
        # DEBUG: index presence
        # -------------------
        _dbg(f"[DBG] Fitting index '{index}'; present in first image?:",
             collection_in.first().bandNames().contains(index))
 
        # 3) Fit harmonic regression: y ~ a0 + a1*sin + a2*cos
        #    (Assumes index band exists in collection_in images)
        reg = collection_fit.select(['t', 'sin', 'cos', index]).reduce(
            ee.Reducer.linearRegression(3, 1)
        )
        coeffs = reg.select('coefficients')
        a0 = coeffs.arrayGet([0, 0])
        a1 = coeffs.arrayGet([0, 1])
        a2 = coeffs.arrayGet([0, 2])
 
        amplitude = a1.pow(2).add(a2.pow(2)).sqrt().rename(f'{index_clean}_amplitude')
        phase = a1.atan2(a2).rename(f'{index_clean}_phase')
 
        # Predict monthly values (avoid toBands to keep names clean)
        months = monthly.toList(monthly.size())
 
        # Predict one month (single-band image with a proper name)
        def predict(i):
            i = ee.Number(i)
            img = ee.Image(months.get(i))
            sin = img.select('sin'); cos = img.select('cos')
            pred = a0.add(a1.multiply(sin)).add(a2.multiply(cos))
            bn = ee.String(index_clean).cat('_m').cat(i.add(1).format('%02d'))
            return pred.rename([bn]).set('system:time_start', img.get('system:time_start'))
 
        # Build the server-side list of predicted images
        predicted_list = ee.List.sequence(0, months.size().subtract(1)).map(predict)
 
        # --- DEBUG the *actual* predicted_list length and first element band names ---
        _dbg(f"[DBG] predicted_list size for '{index_clean}':", predicted_list.size())
        # Only try to print first element band names if size > 0
        first_ok = ee.Number(predicted_list.size()).gt(0)
        first_bands = ee.Algorithms.If(
            first_ok,
            ee.Image(ee.List(predicted_list).get(0)).bandNames(),
            ee.List([])  # empty
        )
        _dbg(f"[DBG] first predicted image band names for '{index_clean}':", first_bands)
 
        # Concatenate predictions into one image without toBands()
        # Start the fold from the first image, then add the rest; avoid an empty start image.
        fitted_image = ee.Image(ee.Algorithms.If(
            ee.Number(predicted_list.size()).gt(0),
            ee.Image(ee.List(predicted_list).slice(1).iterate(
                lambda img, acc: ee.Image(acc).addBands(ee.Image(img)),
                ee.Image(ee.List(predicted_list).get(0))
            )),
            # Fallback placeholder if for some reason no predictions were produced
            ee.Image.constant(0).rename(f'{index_clean}_m00')
        ))
 
 
        # -------------------
        # DEBUG: fitted_image bands
        # -------------------
        _dbg(f"[DBG] Fitted image band names for '{index_clean}':", fitted_image.bandNames())
 
        # SNIC segmentation on the fitted stack
        snic = ee.Algorithms.Image.Segmentation.SNIC(
            fitted_image.clip(roi),
            compactness=30,
            connectivity=8,
            neighborhoodSize=128
        ).select('clusters').rename(f'{index_clean}_snic')
 
        return fitted_image.addBands(amplitude).addBands(phase).addBands(snic)
 
    # Run for all indices and concatenate
    result_images = [fit_index(idx) for idx in index_list]
    out_img = ee.Image.cat(result_images)
 
    # -------------------
    # DEBUG: final image
    # -------------------
    _dbg("[DBG] Final output band count:", out_img.bandNames().size())
   
    _dbg("[DBG] First 10 bands:", out_img.bandNames().slice(0, 10))
 
    return out_img
 
# ============================================
# Cloud Masking
# ============================================
QA_BAND = 'cs_cdf'
CLEAR_THRESHOLD = cloud_threshold
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

def mask_clouds(img):
    img = img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)) \
             .divide(10000).float() \
             .max(0) \
             .copyProperties(img, img.propertyNames())
    return img
img = img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)) \
         .divide(10000).float() \
         .updateMask(img.gte(0)) \
         .copyProperties(img, img.propertyNames())
# ============================================
# Load Sentinel-2 
# ============================================
#Apply Mask
dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(ROI) \
    .filterDate(start, end)

composites = dataset.linkCollection(csPlus, [QA_BAND]) \
    .map(mask_clouds) \
    .select(bands, new_names)\
    .map(lambda img: img.clip(ROI.geometry())) \
    .map(add_indices)


# ============================================
# RUN HARMONICs
# ============================================

harmonic = harmonic_decompose(composites, index_list, start, end, ROI)
def remove_leading_digits_from_band_names(image):
    band_names = image.bandNames()

    # Remove leading numbers and underscores from each band name
    new_band_names = band_names.map(lambda b: ee.String(b).replace('^[0-9]+_', ''))

    return image.rename(new_band_names)
#harmonic_clean = remove_leading_digits_from_band_names(harmonic)
print("Type of harmonic:", type(harmonic))

#harmonic_mosaic = harmonic.mosaic().clip(ROI.geometry())

try:
    ee.data.deleteAsset(s2_asset_id)
    print("ℹ️ Existing asset deleted to allow overwrite.")
except Exception as e:
    print("✅ No existing asset to delete (safe to proceed).")

# Define Export Task
export_task = ee.batch.Export.image.toAsset(
    image=harmonic.clip(ROI),
    description=filename,
    assetId=s2_asset_id,
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

print(f"Pre-processed raster saved to {s2_asset_id}")
print("Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)