import timeit
import time
start_time = timeit.default_timer() 
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import rasterio
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
import joblib
import os
import glob
# ============================================
# Parameters
# ============================================
n_cores = max(1, multiprocessing.cpu_count() - 20)
predictyear = 2020
predictseason = 'A' #NOTE Season A of 2019 started the previous year
sResolution = 10
modelName ='RF'
ntrees = 500
nClasses = 4
classNames = ['Bean', 'Irish Potato', 'Maize', 'Rice']
Class = 'code'
sensor = "S1" 
version = 'v1'
Normalize = False #Normalize data? 
root = '/home/bkenduiywo/'
finalpath = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/'
pcsv_file = f'{root}Data/labels/Nyagatare_{predictseason}{predictyear}.csv'
accuracypath = f'{root}/Classification/accuracy/'
modelfolder = os.path.join(finalpath, 'models')
modelXG = 'XGBoost'
modelRF = 'RF'
# ============================================
# Load current Training Labels  and prepare features and labels
# ============================================
print(f"Loading {pcsv_file} for label prediction!")
# Load into DataFrame
df = pd.read_csv(pcsv_file)
df = df.drop(columns=['system:index', '.geo']) 
print(df.head(n=3))
counts = df[Class].value_counts().sort_values(ascending=True)
print('Distribution of reference data ', counts)
X_test = df.drop(columns=['code'])  # Features
y_true = df['code']  # True labels
# ============================================
# Laod all other trained models 
# ============================================
def load_models(modelName, All=False):
    if(All): #'XG_S1_Multitemp_2020_B_v1.joblib'
        pattern = os.path.join(modelfolder, f'*_{sensor}_Multitemp*.joblib')
        matched_files = glob.glob(pattern)
        # Filter out the excluded file XG_S1_Multitemp_2020_A_v1.joblib
        filtered_files = [f for f in matched_files if not os.path.splitext(os.path.basename(f))[0].endswith(f'{sensor}_Multitemp_{predictyear}_{predictseason}_{version}')]
        print("Model files:", {os.path.basename(f) for f in filtered_files})
        # historical data
        d_files = glob.glob(f'{root}Data/labels/Nyagatare_*.csv') #'/home/bkenduiywo/Data/labels/Nyagatare_B2019.csv'
        d_files = [f for f in d_files if not os.path.splitext(os.path.basename(f))[0].endswith(f'Nyagatare_{predictseason}{predictyear}')]
        df = pd.concat([pd.read_csv(f) for f in d_files], ignore_index=True)
        df = df.drop(['system:index', '.geo'], axis=1)
    else:
        pattern = os.path.join(modelfolder, f'{modelName}_{sensor}_Multitemp*.joblib')
        matched_files = glob.glob(pattern)
        # Filter out the excluded file XG_S1_Multitemp_2020_A_v1.joblib
        filtered_files = [f for f in matched_files if not f.endswith(f'{modelName}_{sensor}_Multitemp_{predictyear}_{predictseason}_{version}.joblib')]
        print("Model files:", {os.path.basename(f) for f in filtered_files}) 
        # historical data
        d_files = glob.glob(f'{root}Data/labels/Nyagatare_*.csv') #'/home/bkenduiywo/Data/labels/Nyagatare_B2019.csv'
        d_files = [f for f in d_files if not os.path.splitext(os.path.basename(f))[0].endswith(f'Nyagatare_{predictseason}{predictyear}')]
        df = pd.concat([pd.read_csv(f) for f in d_files], ignore_index=True)
        df = df.drop(['system:index', '.geo'], axis=1)
    # Load models and label as tuples (e.g., ('XG_1', <model object>))
    XG_models = []
    for i, filepath in enumerate(filtered_files, start=1):
        model = joblib.load(filepath)
        label = f'{modelName}_{i}'
        XG_models.append((label, model))
    return XG_models, df
XG_models, df_hist = load_models('XG')
RF_models, df_hist = load_models('RF')
ALL_models, df_hist = load_models('All', All=True)
X_hist = df_hist.drop(columns=['code'])  # Features
y_hist = df_hist['code']  # True labels
'''
# ============================================
# Transfer models without  current labels
# ============================================
from scipy.stats import mode
from collections import Counter
def majority_vote(models, X):
    predictions = []
    for label, model in models:
        # Ensure the input features match the training feature order
        if hasattr(model, 'feature_names_in_'):
            X_aligned = X[model.feature_names_in_]
        else:
            raise ValueError(f"Model {label} does not have 'feature_names_in_' attribute.")
        
        preds = model.predict(X_aligned)
        predictions.append(preds)
    
    predictions = np.array(predictions)  # shape: (n_models, n_samples)
    final_preds, _ = mode(predictions, axis=0)
    return final_preds.flatten()

# Make sure X_test includes all necessary columns
All_pred_maj = majority_vote(ALL_models, X_test)
print("🧮 Predicted class distribution:", Counter(All_pred_maj))
RF_pred_maj = majority_vote(RF_models, X_test)
print("🧮 Predicted class distribution:", Counter(RF_pred_maj))
XG_pred_maj = majority_vote(XG_models, X_test)
print("🧮 Predicted class distribution:", Counter(XG_pred_maj))
'''
# ============================================
# Training ML Models
# ============================================
def ml_ensemble(models, modelname):
    base_estimators =  [models] # List of (name, model) tuples
    # Initialize StackingClassifier: Use Logistic Regression as the final estimator
    final_estimator = LogisticRegression(multi_class='multinomial', random_state=123)
    stacking_clf = StackingClassifier(
       estimators=XG_models,
       final_estimator=final_estimator,
       cv=5,  # Cross-validation for meta-model training
       passthrough=False,  # Use only base model predictions (not original features)
       n_jobs=n_cores  # Use all available cores
    )
    # 6. Fit StackingClassifier on test data: Note: In practice, fit on a separate validation set to avoid overfitting
    #stacking_clf.fit(X_test, y_true)
    print(f'Completed stacking classifier for {modelname}')
    return stacking_clf

XG_en = ml_ensemble(XG_models, 'XGboost ensemble')
XG_en.fit(X_hist, y_hist)
RF_en = ml_ensemble(RF_models, 'RF ensemble')
RF_en.fit(X_hist, y_hist)
All_en = ml_ensemble(ALL_models, 'RF and XGboost ensemble')
All_en.fit(X_hist, y_hist)
# ============================================
# Ensemble prediction
# ============================================
XG_pred = XG_en.predict(X_test)
RF_pred = RF_en.predict(X_test)
All_pred = All_en.predict(X_test)
# ============================================
# Evaluate performance
# ============================================
# F1-score (weighted average for multi-class)
def compute_metrics(predictions):
    f1 = f1_score(y_true, predictions, average=None, labels=list(range(nClasses)))
    for label, score in zip(list(range(nClasses)), f1):
        print(f"F1-Score for {classNames[label]} (class {label}): {score:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, predictions, labels=list(range(nClasses)))
    print("\nConfusion Matrix:")
    print(cm)
    # Overall accuracy
    accuracy = accuracy_score(y_true, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, predictions)
    print(f"Cohen's Kappa: {kappa:.4f}")
    # Optional: Pretty-print confusion matrix with labels
    print("\nConfusion Matrix (Labeled):")
    cm_df = pd.DataFrame(cm,
        index=[f"True {label}" for label in classNames],
        columns=[f"Pred {label}" for label in classNames]
    )
    print(cm_df)
    return f1, cm

XG_f1, XG_cm = compute_metrics(XG_pred)
RF_f1, RF_cm = compute_metrics(RF_pred)
# ============================================
# Save evaluation metrics
# ============================================

metrics_file = f"{accuracypath}{sensor}_{predictyear}_{predictseason}_{modelXG}_ensemble_accuracy.xlsx"

def save_metrics(filename):
    # 1. F1-scores
    f1_dict = {f"F1: {classNames[label]}": [score] for label, score in zip(range(nClasses), f1)}
    # 2. Overall Accuracy and Kappa
    overall_metrics = {
        "Overall_Accuracy": [accuracy],
        "Cohens_Kappa": [kappa]
    }
    # Merge F1 and overall metrics into one DataFrame
    metrics_df = pd.DataFrame({**f1_dict, **overall_metrics})
    
    # 3. Save confusion matrix separately with class labels
    cm_df = pd.DataFrame(
        cm,
        index=[f"{label}" for label in classNames],
        columns=[f"{label}" for label in classNames]
    )

    # 4. Save to a single Excel file with two sheets
    with pd.ExcelWriter(filename) as writer:
        metrics_df.to_excel(writer, sheet_name="Summary_Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        print(f"Metrics and confusion matrix saved to {filename}.")
   
save_metrics(metrics_file)

###PLots
# Ensure seaborn style
sns.set(style="whitegrid", font_scale=1.1)

# === 1. Confusion Matrix Plot ===
def plot_confusion_matrix_percent(cm, classNames, title='Confusion Matrix (%)', cmap='Blues'):
    # Normalize per row (true class) to get percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=cmap,
                     xticklabels=classNames, yticklabels=classNames,
                     linewidths=0.5, linecolor='gray', cbar=True)
    
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# === 2. F1 Score Bar Plot ===
def plot_f1_scores(f1, classNames, title='F1 Scores per Class'):
    plt.figure(figsize=(8, 4))
    bars = sns.barplot(x=classNames, y=f1, palette='viridis')
    plt.ylim(0, 1.05)
    plt.ylabel('F1 Score')
    plt.title(title)

    # Annotate each bar with the F1 score value
    for bar, score in zip(bars.patches, f1):
        bars.annotate(f"{score:.2f}", 
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()



plot_f1_scores(xg_f1, classNames)
plot_confusion_matrix_percent(xg_cm, classNames)

print(f" Validation for {predictyear} season {predictseason} complete!")
print("Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)