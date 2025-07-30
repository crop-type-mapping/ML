import ee
import geemap
import geopandas as gpd

# Authenticate and initialize Earth Engine
service_account = 'your-service-account@your-project.iam.gserviceaccount.com'
key_path = 'path-to-your-private-key.json'
credentials = ee.ServiceAccountCredentials(service_account, key_path)
ee.Initialize(credentials)

# Load training polygons from a shapefile
shapefile_path = 'training_polygons.shp'  # path to your local shapefile
gdf = gpd.read_file(shapefile_path)

# Upload shapefile to GEE as asset beforehand or convert it to FeatureCollection in-memory
fc = geemap.gdf_to_ee(gdf)

# Define time periods
past_year = '2022'
target_year = '2025'
start_month = 3
end_month = 4

# Define Sentinel-2 bands to correct
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

# Function to get Sentinel-2 seasonal composite
def get_s2_composite(year):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(f'{year}-{start_month:02d}-01', f'{year}-{end_month:02d}-30') \
        .filterBounds(fc.geometry()) \
        .map(lambda img: img.select(bands).multiply(0.0001).copyProperties(img, img.propertyNames()))
    return s2.median().set('system:time_start', ee.Date(f'{year}-{start_month:02d}-01').millis())

# Function to get seasonal CHIRPS rainfall and CHIRTSmax temperature mean
def get_climate_composite(year):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(f'{year}-{start_month:02d}-01', f'{year}-{end_month:02d}-30') \
        .filterBounds(fc.geometry()).select('precipitation')

    chirts = ee.ImageCollection("IDAHO_EPSCOR/CHIRTS/daily") \
        .filterDate(f'{year}-{start_month:02d}-01', f'{year}-{end_month:02d}-30') \
        .filterBounds(fc.geometry()).select('Tmax')

    return chirps.mean().rename('rain') \
        .addBands(chirts.mean().rename('temp'))

# Load data
past_img = get_s2_composite(past_year)
current_img = get_s2_composite(target_year)
climate_past = get_climate_composite(past_year)
climate_current = get_climate_composite(target_year)

# Compute climate anomaly: current - past
climate_diff = climate_current.subtract(climate_past)

# Stack bands and climate diff for regression input
past_stack = past_img.addBands(climate_diff)

# Sample training points from polygons
training = past_stack.sampleRegions(
    collection=fc,
    properties=['code'],
    scale=10,
    tileScale=4
)

# Train regression model per band
trained_models = {}
predicted_bands = []
for band in bands:
    rf = ee.Classifier.smileRandomForest(numberOfTrees=100).setOutputMode('REGRESSION')
    trained = rf.train(
        features=training,
        classProperty=band,
        inputProperties=['rain', 'temp']
    )
    prediction = climate_diff.classify(trained).rename(band + '_delta')
    predicted_bands.append(prediction)

# Stack predicted corrections and subtract from past image
corrections = ee.Image(predicted_bands)
corrected_image = past_img.select(bands).subtract(corrections).rename([b + '_corrected' for b in bands])

# Export or display
Map = geemap.Map()
Map.centerObject(fc, 10)
Map.addLayer(corrected_image, {"bands": ['B4_corrected', 'B3_corrected', 'B2_corrected'], "min": 0, "max": 0.3}, 'Corrected 2022')
Map
