#============================== Import Libraries ==============================#
import timeit
start_time = timeit.default_timer() #Time the process

from sklearnex import patch_sklearn #import scikit-learn after these lines. see https://stackoverflow.com/questions/53940258/svc-classifier-taking-too-much-time-for-training
patch_sklearn() #import scikit-learn after these lines.
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import joblib
import multiprocessing
import pandas as pd
from rasterio.merge import merge
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.simplefilter("ignore", category=UserWarning)
print('Neccessary libraries imported!')

#============================== Global variables ==============================#
rawpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/RAW/' 
interimpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/Process/'
finalpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/'
resultspath = '/home/bkenduiywo/Classification/'
trainpath = '/home/bkenduiywo/shapefiles/'
#testimpath = os.path.join(interimpath, 'S2_2021_B_image_chunk_0.tif')
pfigures = '/home/bkenduiywo/Figures/'
print('Global variables initialized!')
sResolution = 10 #Spatial resolution
sensor = "S1" #S1_Multitemp_B2025_mosaic.tif
season ='B'
eyear  = '2025';
nClasses = 4 # Number of lancover classes (Ensure to start numbering from zero)
n_classes = nClasses
Class = "code"
ntrees = 200
modelName ='RF'
version ='v1'
ignore_value = 255
no_data = -9999
n_cores = max(1, multiprocessing.cpu_count() - 10)
print(f" Using {n_cores} parallel workers.")
class_names = ['Bean', 'Irish Potato', 'Maize', 'Rice'] #['Banana', 'Bean', 'Irish Potato', 'Maize', 'Rice', 'Sorghum', 'Soybean']

from sklearn.preprocessing import MinMaxScaler
s1_path = os.path.join(finalpath, f"images/{sensor}_Multitemp_{season}{eyear}_mosaic.tif")
label_raster_path  = os.path.join(trainpath, 'RWA_B2025_Merge_v2.tif')
# ==========prepare data for training pixels extraction================= #
print("Extracting training samples from rasterized labels...")

with rasterio.open(s1_path) as src:
    band_names = src.descriptions
    image_data = src.read().reshape(src.count, -1).T
    nodata_feature = src.nodata
print('No data value in Satellite image:', {nodata_feature}, flush=True)

with rasterio.open(label_raster_path) as lbl_src:
    labels = lbl_src.read(1).flatten()
    nodata_label = lbl_src.nodata if lbl_src.nodata is not None else ignore_value
print('No data value in label image:', {nodata_label}, flush=True)

mask = (labels != nodata_label) & np.isin(labels, valid_classes)
X_train = image_data[mask]
y_train = labels[mask]

print(f"Extracted {X_train.shape[0]} valid training pixels.")

training_data = pd.DataFrame(X_train, columns=band_names)
training_data[Class] = y_train

#============================== Model Training ==============================#
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
# ✅ Random Forest
rf_model = RandomForestClassifier(n_estimators=ntrees, n_jobs=n_cores, random_state=123)
#stacked_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
print('CompletedFitting RF model!')
#===
# Feature importance
feature_importance = rf_model.feature_importances_
feature_names = [f"Feature {i+1}" for i in range(X_train.shape[1])]
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print('Feature importance: ',importance_df)

# Define output CSV path (same directory as plot)
csv_path = os.path.join(resultspath, f"{sensor}_{eyear}_{season}_{version}_RF_importance.csv")
# Save to CSV
importance_df.to_csv(csv_path, index=False)
print(f"✅ Random Forest feature importances saved to {csv_path}")

#Export plot 

# 📁 Define the path where to save the plot
plot_path = os.path.join(resultspath, f"{sensor}_{eyear}_{season}_{version}_RF_importance.png")

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

print(f"✅ Feature importance plot saved to: {plot_path}")

# 🎯 Select only important bands > 5%
s2bands = importance_df.query("Importance > 5")['Feature'].to_list()
print('Selected bands:', s2bands)

#====

modelfolder = os.path.join(finalpath, 'models')
os.makedirs(modelfolder, exist_ok=True)
modelpath = os.path.join(modelfolder, f"{modelName}_{sensor}_{eyear}_{season}_{version}.joblib") #rf_model_2021_B.joblib {version}
import joblib
joblib.dump(rf_model, modelpath)
print('Fitted a model saved to: ', modelpath)


print("\n Done!")
print("Elapsed time hours: ", (timeit.default_timer() - start_time)/3600.0)

#####



