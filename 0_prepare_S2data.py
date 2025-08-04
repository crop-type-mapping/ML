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
from google.oauth2 import service_account
import timeit
import time
import geemap
import re

start_time = timeit.default_timer()

# ============================================
# Initialize Earth Engine
# ============================================
SERVICE_ACCOUNT_KEY = "gee_key.json"
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_KEY,
    scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/drive']
)
ee.Initialize(credentials)
print("✅ Earth Engine initialized successfully!")

# ============================================
# Parameters
# ============================================
eyear = 2025
start = ee.Date(f'{eyear}-02-01')
end = ee.Date(f'{eyear}-06-30')
season = 'B'
version = 'v1'
cloud_threshold = 0.50
sResolution = 10

filename = f'S2_Composite_{eyear}_{season}_{version}'
gee_project = "projects/cropmapping-365811"
asset_folder = "rwanda"
s2_asset_id = f"{gee_project}/assets/{asset_folder}/{filename}"

bands = ['B2', 'B3', 'B4', 'B7', 'B8', 'B11', 'B12']
new_names = ['blue', 'green', 'red', 're3', 'nir', 'swir1', 'swir2']
index_list = ['cvi', 'gvmi', 'ndvi', 'ndmi']

# ============================================
# Load ROI
# ============================================
cList = ['Nyagatare','Nyabihu','Ruhango','Musanze']
ROI = ee.FeatureCollection('users/bensonkemboi/CIAT/Rwanda/rwa_adm2_selected_districts')
ROI = ROI.filter(ee.Filter.inList('ADM2_EN', cList))

Map = geemap.Map(center=[-1.94, 29.87], zoom=8)
Map.addLayer(ROI, {}, 'ROI')

# ============================================
# Spectral Indices
# ============================================
def add_indices(image):
    b = {
        'blue': image.select('blue'),
        'green': image.select('green'),
        'red': image.select('red'),
        'nir': image.select('nir'),
        'swir1': image.select('swir1'),
        'swir2': image.select('swir2'),
        're3': image.select('re3'),
    }
    cvi = image.expression('(nir/green)*(red+green)', b)
    gvmi = image.expression('((re3+0.1)-(swir2+0.02))/((re3+0.1)+(swir2+0.02))', b)
    ndvi = image.expression('(nir-red)/(nir+red)', b)
    ndmi = image.expression('(nir - swir1)/(nir + swir1)', b)
    return image.addBands(ee.Image.cat([cvi, gvmi, ndvi, ndmi]).rename(['cvi','gvmi','ndvi','ndmi']).float())

# ============================================
# Add Time Bands
# ============================================
def add_time(image):
    date = ee.Date(image.get('system:time_start'))
    days = date.difference(ee.Date.fromYMD(eyear, 1, 1), 'day')
    radians = days.multiply(2 * 3.1416 / 365.25)
    return image.addBands([
        ee.Image.constant(days).rename('t').toFloat(),
        ee.Image.constant(radians.sin()).rename('sin').toFloat(),
        ee.Image.constant(radians.cos()).rename('cos').toFloat()
    ], overwrite=True)

# ============================================
# Cloud Masking
# ============================================
QA_BAND = 'cs_cdf'
CLEAR_THRESHOLD = cloud_threshold
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

def mask_clouds(img):
    img = img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)) \
             .divide(10000).float() \
             .updateMask(img.gte(0)) \
             .copyProperties(img, img.propertyNames())
    return img

# ============================================
# Load Sentinel-2 and Preprocess
# ============================================
dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(ROI) \
    .filterDate(start, end)

composites = dataset.linkCollection(csPlus, [QA_BAND]) \
    .map(mask_clouds) \
    .select(bands, new_names) \
    .map(lambda img: img.clip(ROI.geometry())) \
    .map(add_indices)

# ============================================
# Debug helper
# ============================================
def _dbg(msg, value):
    try:
        print(msg, value.getInfo())
    except Exception as e:
        print(msg, f"(getInfo failed: {e})")

# ============================================
# Seasonal Harmonic Decomposition
# ============================================
def harmonic_decompose_seasonal(collection_in, index_list, start, end, roi):
    _dbg("[DBG] Input collection size:", collection_in.size())
    _dbg("[DBG] Input first image bands:", collection_in.first().bandNames())

    collection_fit = collection_in.map(add_time)

    def fit_index(index):
        index_clean = re.sub(r'^[^A-Za-z]+', '', index)

        reg = collection_fit.select(['t', 'sin', 'cos', index]).reduce(
            ee.Reducer.linearRegression(3, 1)
        )
        coeffs = reg.select('coefficients')
        a0 = coeffs.arrayGet([0,0])
        a1 = coeffs.arrayGet([0,1])
        a2 = coeffs.arrayGet([0,2])

        amplitude = a1.pow(2).add(a2.pow(2)).sqrt().rename(f'{index_clean}_amplitude')
        phase = a1.atan2(a2).rename(f'{index_clean}_phase')

        # Mean sin/cos for the season
        sin_mean = collection_fit.select('sin').mean()
        cos_mean = collection_fit.select('cos').mean()

        pred = a0.add(a1.multiply(sin_mean)).add(a2.multiply(cos_mean)).rename(f'{index_clean}_fitted')

        return pred.addBands(amplitude).addBands(phase)

    fitted_seasonal = ee.Image.cat([fit_index(idx) for idx in index_list])

    snic = ee.Algorithms.Image.Segmentation.SNIC(
        fitted_seasonal.clip(roi),
        compactness=30,
        connectivity=8,
        neighborhoodSize=128
    ).select('clusters').rename('s2_obia')

    return fitted_seasonal.addBands(snic)

# ============================================
# Run Seasonal Harmonics
# ============================================
harmonic = harmonic_decompose_seasonal(composites, index_list, start, end, ROI)

# ============================================
# Export to Asset
# ============================================
try:
    ee.data.deleteAsset(s2_asset_id)
    print("Existing asset deleted to allow overwrite.")
except Exception:
    print("No existing asset to delete (safe to proceed).")

export_task = ee.batch.Export.image.toAsset(
    image=harmonic.clip(ROI),
    description=filename,
    assetId=s2_asset_id,
    region=ROI.geometry(),
    scale=sResolution,
    crs='EPSG:4326',
    maxPixels=1e13
)
export_task.start()
print(f"Export task '{filename}' started...")

while export_task.active():
    print("Export in progress... waiting 30s...")
    time.sleep(30)

status = export_task.status()
print(f"Export task completed with status: {status['state']}")
if status['state'] == 'FAILED':
    print(f"Error message: {status.get('error_message','No error message provided.')}")
print(f"Pre-processed raster saved to {s2_asset_id}")
print("Elapsed time (hours):", (timeit.default_timer() - start_time)/3600)
