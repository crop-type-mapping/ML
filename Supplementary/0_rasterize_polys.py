# ------------------------------------------------------------
# Author: Benson Kenduiywo
# Rasterize polygon labels  
# ------------------------------------------------------------
import geopandas as gpd
import os
import rasterio
from rasterio.features import rasterize

# -----------------------
# Paths
# -----------------------
sensor = "S1" 
season ='B'
eyear  = '2025'
root  = '/home/bkenduiywo/shapefiles/'
finalpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/'
sentinel_raster_path = os.path.join(finalpath, f"images/{sensor}_Multitemp_{season}{eyear}_mosaic.tif")
vector_path = f'{root}RWA_B2025_Merge_v2.shp'
reference_raster_path = os.path.join(
    root, os.path.splitext(os.path.basename(vector_path))[0] + ".tif"
)

# -----------------------
# Read reference raster metadata
# -----------------------
with rasterio.open(sentinel_raster_path) as ref:
    crs = ref.crs
    transform = ref.transform
    height = ref.height
    width = ref.width
    nodata_value = 255

print("Rasterizing reference polygons...", flush=True)

# -----------------------
# Read and filter vector data
# -----------------------
gdf = gpd.read_file(vector_path).to_crs(crs)
gdf = gdf[gdf.geometry.notnull()]

valid_classes = [0, 1, 2, 3]  # Only these classes will be rasterized
gdf = gdf[gdf["code"].isin(valid_classes)]

shapes = [(geom, int(code)) for geom, code in zip(gdf.geometry, gdf["code"])]

# -----------------------
# Rasterize
# -----------------------
ref_rasterized = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=nodata_value,
    dtype="uint8"
)

# -----------------------
# Save raster
# -----------------------
profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": crs,
    "transform": transform,
    "nodata": nodata_value,
    "compress": "DEFLATE",
    "tiled": True
}

with rasterio.open(reference_raster_path, "w", **profile) as dst:
    dst.write(ref_rasterized, 1)

'''
import matplotlib.patches as mpatches
# -----------------------
# Plot raster with legend
# -----------------------
cmap = plt.get_cmap("tab10", len(valid_classes))  # distinct colors
masked_raster = np.ma.masked_equal(ref_rasterized, nodata_value)

plt.figure(figsize=(10, 8))
im = plt.imshow(masked_raster, cmap=cmap, vmin=0, vmax=len(valid_classes)-1)
plt.title("Rasterized Reference Polygons", fontsize=14)
plt.axis("off")

# Create legend patches
valid_names =['Bean', 'Irish Potatot', 'Maize', 'Rice']
legend_patches = [mpatches.Patch(color=cmap(i), label=f"{cls}") 
                  for i, cls in enumerate(valid_names)]
plt.legend(handles=legend_patches, loc="upper right", fontsize=10)

plt.show()
'''
print(f"Reference raster saved: {reference_raster_path}", flush=True)



