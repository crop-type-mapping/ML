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
import geemap
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

from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
# Initialize Earth Engine
geemap.ee_initialize()
print('GeeMap initialized!')

# ============================================
# Parameters
# ============================================
n_cores = max(1, multiprocessing.cpu_count() - 20)
syear = 2021
eyear = 2021
season = 'B' #NOTE Season A of 2019 started the previous year
start = ee.Date(f'{syear}-02-01')
end = ee.Date(f'{eyear}-06-30')
targetstart = ee.Date(f'2025-02-01')
targetend = ee.Date(f'2025-06-30')
timeField = 'system:time_start'
sResolution = 10
modelName ='RF'
ntrees = 500
Class = 'code'
sensor = "S1" 
version = 'v1'
Normalize = False #Normalize data? 
root = '/home/bkenduiywo/Data/'
trainData = f'{root}shp/Nyagatare_{season}{eyear}.tif'
filename = f'{sensor}_Multitemp_{eyear}_{season}_{version}'
gee_project = "projects/cropmapping-365811"
asset_folder = "rwanda"
s1_asset_id = f"{gee_project}/assets/{asset_folder}/{filename}"
# Set asset ID (replace with your actual path)
resultspath = '/home/bkenduiywo/Classification/'
finalpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/'
fc_filename = os.path.basename(trainData)
fc_filename = os.path.splitext(fc_filename)[0]
fc_asset_id = f"{gee_project}/assets/{asset_folder}/{fc_filename}_traingsites"
csv_file = f'{root}labels/{fc_filename}.csv'
# ============================================
# Load Training Labels 
# ============================================
with rasterio.open(trainData) as src:
    label_array = src.read(1)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds

# Convert to points (i, j) for non-NaN/non-255 pixels
rows, cols = np.where((label_array >= 0) & (label_array <= 3))
codes = label_array[rows, cols]

# Convert row/col to lon/lat
xs, ys = rasterio.transform.xy(transform, rows, cols)
points = gpd.GeoDataFrame({'code': codes}, geometry=gpd.points_from_xy(xs, ys), crs=crs)

# Reproject to EPSG:4326 for Earth Engine
points = points.to_crs(epsg=4326)

# Convert to Earth Engine FeatureCollection
features = [
    ee.Feature(ee.Geometry.Point(xy.x, xy.y), {'code': int(c)})
    for xy, c in zip(points.geometry, points.code)
]
train_fc = ee.FeatureCollection(features)

# Group features by 'code' and count
code_counts = train_fc.aggregate_histogram('code').getInfo()

# Display the counts
print("Label distribution in train_fc:")
for label, count in sorted(code_counts.items()):
    print(f"  Class {label}: {count} points")
# ============================================
# Load ROI 
# ============================================
# load Training data
ROI = ee.FeatureCollection('users/bensonkemboi/CIAT/Rwanda/rwa_adm2_selected_districts')
#Map = geemap.Map(center=[-1.94, 29.87], zoom=8)
#Map.addLayer(ROI, {}, 'ROI')

# ============================================
# Sentinel-1 images
# ============================================
s1Base = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(train_fc) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .filter(ee.Filter.notNull(['system:time_start'])) \
    .map(lambda img: img.clip(ROI))
# ============================================
# CHIRPS & ERA5
# ============================================
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterBounds(ROI) \
    .map(lambda img: img.clip(train_fc))

era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
    .filterDate(start, end) \
    .filterBounds(ROI) \
    .map(lambda img: img.clip(train_fc))
target_era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
    .filterDate(targetstart, targetend) \
    .filterBounds(ROI) \
    .map(lambda img: img.clip(train_fc))
# Function to extract and rename selected variables
def extract_vars(img):
    return img.select([
        #'dewpoint_temperature_2m',
        #'temperature_2m_max',
        #'temperature_2m',
        #'surface_pressure',
        #'surface_solar_radiation_downwards_sum',
        'total_precipitation_sum'
    ], [
         'TPS'
    ])

# Apply to both periods
varsBase = era5.map(extract_vars)
varsTarget = target_era5.map(extract_vars)
# Function to compute ClimI image
def computeClimI_Image(imageCollection, cumulativeDays):
    #varNames = ['DT', 'TMa', 'TMe', 'SP', 'SSRDS', 'TPS']
    varNames = ['TPS']

    def smooth_band(band_name):
        filtered = imageCollection.select(band_name)
        dates = filtered.aggregate_array('system:time_start')

        def smooth_for_date(d):
            d = ee.Date(d)
            window = filtered.filterDate(
                d.advance(-cumulativeDays / 2, 'day'),
                d.advance(cumulativeDays / 2, 'day')
            )
            return window.mean().set('system:time_start', d.millis())

        smoothed = ee.ImageCollection(dates.map(smooth_for_date))
        return smoothed.mean().rename(band_name)

    smoothedImages = [smooth_band(b) for b in varNames]
    return ee.Image.cat(smoothedImages).clip(train_fc)

# Step 2: Compute seasonal ClimI images
climI_base = computeClimI_Image(varsBase, 5)
climI_target = computeClimI_Image(varsTarget, 5)

# Step 3: Compute relative difference  #.divide(climI_base) \
relDiff = climI_target.subtract(climI_base) \
                        .rename(['TPS'])

# Step 4: Compute L2 norm (climate distance)
climID_L2 = relDiff.rename('climID_L2')#.pow(2).reduce(ee.Reducer.sum()).sqrt().rename('climID_L2')
# Get min and max climID_L2 values in the region
stats = climID_L2.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=train_fc.geometry(),
    scale=1000,
    maxPixels=1e13
).getInfo()

print(f"Minimum climID_L2: {stats['climID_L2_min']:.4f}")
print(f"Maximum climID_L2: {stats['climID_L2_max']:.4f}")
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
              .addBands(ee.Image.constant(1)) #\
              #.updateMask(datamask)
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
            .map(addVariables).map(maskBorderNoise) \
            .map(refinedLee).map(toGamma0).map(addRadarIndices)
        s1_comp = s1_monthly.median()
        s1_proj = s1_comp.select('VV').projection()

        rain = chirps.filterDate(month_start, month_end).select('precipitation').sum() \
            .reproject(crs=s1_proj.crs(), scale=sResolution).resample('bilinear').rename('ch_rain')
        evap = era5.select('total_evaporation_sum').mean() \
            .reproject(crs=s1_proj.crs(), scale=1000).resample('bilinear').rename('era_evap')
        temp = era5.select('temperature_2m').mean().subtract(273.15) \
            .reproject(crs=s1_proj.crs(), scale=1000).resample('bilinear').rename('era_t2m')

        combined = s1_comp.addBands([rain, evap, temp]) \
            .select(['VV', 'VH', 'cri', 'mrfdi', 'rvi', 'ch_rain', 'era_t2m']) \
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

        return combined.addBands(s1_obia)

    return months.map(monthly_fn)

# Create monthly ImageCollection
monthlyComposites = ee.ImageCollection.fromImages(get_monthly_composites(start, end))
first_image = monthlyComposites.first()
band_names = first_image.bandNames().getInfo()
print('Monthly composites (S1 + Rain + Temp):', band_names)

# ============================================
# Training pixels extraction
# ============================================
# Define bands from ImageCollection (use all bands from first image)
bands = ee.Image(monthlyComposites.first()).bandNames().getInfo()
print('Bands: ', bands)
# Sample all pixels within training polygons from each image

def sample_image(img):
    return img.select(bands).sampleRegions(
        collection=train_fc,
        properties=['code'],
        scale=10,
        geometries=True
    )

# Merge all samples from the image collection
sample_fc = monthlyComposites.map(sample_image).flatten()
# Filter out features with null geometries
sample_fc = sample_fc.filter(ee.Filter.notNull(['.geo']))
print('Training pixels extraction complete!')

#=============Save training data as asset======

# Delete existing asset if exists (overwrite support)
try:
    ee.data.deleteAsset(fc_asset_id)
    print("ℹ️ Existing asset deleted to allow overwrite.")
except Exception as e:
    print("✅ No existing asset to delete (safe to proceed).")

# Start export to asset
task = ee.batch.Export.table.toAsset(
    collection=sample_fc,
    description='Export_Training_Samples_Asset',
    assetId=fc_asset_id
)
task.start()

# Monitor task
print("📤 Exporting FeatureCollection to Earth Engine Asset...")
while task.active():
    print("⏳ Waiting for export to complete...")
    time.sleep(30)

status = task.status()
print(f"✅ Export task completed with status: {status['state']}")
print(f"Training sites saved to {fc_asset_id}")
if status['state'] == 'FAILED':
    print(f"❌ Error: {status.get('error_message', 'No error message')}")

#=============Save training data to disk ======    
# Request download URL (CSV format)
asset_fc = ee.FeatureCollection(fc_asset_id)
# Request download URL (CSV format)
url = asset_fc.getDownloadURL(filetype='CSV')

# Download the CSV
import requests
response = requests.get(url)
with open(csv_file, 'wb') as f:
    f.write(response.content)

print(f"✅ CSV saved to {csv_file}")
# Load into DataFrame
df = pd.read_csv(csv_file)
print(df.head(n=3))
counts = df[Class].value_counts().sort_values(ascending=True)
print('Distribution of reference data ', counts)
# ============================================
# Training ML Models
# ============================================
sampled_df = df
# Remove NA values
sampled_df.dropna(inplace=True)

#===Optional normalize each band between 0 and 1
if Normalize:
    print("Normalization is enabled.")
    # Separate features and label
    features = sampled_df.drop(columns=[Class,'system:index', '.geo'])  # all columns except 'code'
    labels = sampled_df['code']                   # label column
    
    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    # Reattach label column as the last column
    sampled_df = features_scaled.copy()
    sampled_df['code'] = labels
    
    print("\n✅ Normalized data preview:")
    print(sampled_df.head(3))

#========Show Sample Data ===================#
X_train = sampled_df.drop(columns=[Class,'system:index', '.geo']) 
print('Created predictors successfully, i.e.:\n', X_train.head(n=3))
y_train = sampled_df[Class]
print('Created labels successfully, i.e.:\n', y_train.head(n=3))

#============================== Model Training ==============================#
warnings.filterwarnings("ignore")
#Random Forest
rf_model = RandomForestClassifier(n_estimators=ntrees, n_jobs=n_cores, random_state=123)
# XGBoost
xg_model = XGBClassifier(n_estimators=ntrees, eval_metric='mlogloss', random_state=123, n_jobs=-n_cores)
#stacked_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xg_model.fit(X_train, y_train)
print('CompletedFitting RF model!')
#===
# Feature importance
feature_importance = rf_model.feature_importances_
feature_names = [bands[i] for i in range(X_train.shape[1])]
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print('Feature importance: ',importance_df)

# Define output CSV path (same directory as plot)
csv_path = os.path.join(resultspath, f"{filename}_RF_importance.csv")
# Save to CSV
importance_df.to_csv(csv_path, index=False)
print(f"Random Forest feature importances saved to {csv_path}")

#Export plot 

# 📁 Define the path where to save the plot
plot_path = os.path.join(resultspath, f"{filename}_RF_importance.png")

# 📊 Create the barplot
plt.figure(figsize=(12, 8))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    palette="viridis",
    edgecolor="black"
)
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

# 🔤 Labels and Title
plt.xlabel('Importance (%)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title(f"{sensor} {eyear} Season {season} Feature Importance", fontsize=16, fontweight='bold')

# 🔄 Invert Y-axis for descending importance
plt.gca().invert_yaxis()

# 💾 Save to disk before displaying
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Feature importance plot saved to: {plot_path}")

#========Save trained model ===================#
modelfolder = os.path.join(finalpath, 'models')
os.makedirs(modelfolder, exist_ok=True)
modelpath = os.path.join(modelfolder, f"{modelName}_{filename}.joblib") #rf_model_2021_B.joblib {version}
xg_modelpath = os.path.join(modelfolder, f"XG_{filename}.joblib")
import joblib
joblib.dump(rf_model, modelpath)
joblib.dump(xg_model, xg_modelpath)
print('Fitted RF model saved to: ', modelpath)
print('Fitted Xgboost model saved to: ', xg_modelpath)

print(f" Model training complete")
print("Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)