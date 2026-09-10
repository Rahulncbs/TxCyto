# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: python/3.9
#     language: python
#     name: py3.9
# ---

# +
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import zscore
import seaborn as sns
# -

df_tcga=pd.read_csv("/data/kumarr17/merged_sample_mtx/CD14_Merged_mRNA_Expression.tsv", index_col=0)
df_tcga_v1=pd.read_csv("/data/kumarr17/merged_sample_mtx/TCGA_Merged_mRNA_Expression.tsv", index_col=0)

# +
cyt_of_interest=list(pd.read_csv("/data/kumarr17/supplemetary_table_cyt_manuscript/cytokine_list.csv",index_col='Unnamed: 0')['cytokine']) ## list of cytokine to train model

## Feature vector
with open("/data/kumarr17/common_features_ver1.txt", "r") as f:
    model_feature_list = [line.strip() for line in f]

filtered_model_feature_list = [x for x in model_feature_list if x not in set(cyt_of_interest)]
# -

df_tcga_features=df_tcga[filtered_model_feature_list]
df_tcga_cyt_col=df_tcga_v1[cyt_of_interest]

df_tcga_features = df_tcga_features[~df_tcga_features.index.duplicated(keep='first')]
df_tcga_cyt_col = df_tcga_cyt_col[~df_tcga_cyt_col.index.duplicated(keep='first')]

# +
# Find common rows
common_rows = df_tcga_features.index.intersection(df_tcga_cyt_col.index)

# Keep only common rows in both
df_tcga_features = df_tcga_features.loc[common_rows]
df_tcga_cyt_col = df_tcga_cyt_col.loc[common_rows]

# +
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# speed settings
# -----------------------------
tf.keras.backend.clear_session()

cyt_tmp = cyt_of_interest
corr_new_tmp = []

# features
X_raw = df_tcga_features.values.astype("float32")

# optional: row-wise z-score, faster than apply(zscore, axis=1)
X_raw = (X_raw - X_raw.mean(axis=1, keepdims=True)) / (
    X_raw.std(axis=1, keepdims=True) + 1e-8
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def build_fcnn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

for cytokine_to_check in cyt_tmp:
    
    print(f"\n==============================")
    print(f"Cytokine: {cytokine_to_check}")
    print(f"==============================")
    
    y = df_tcga_cyt_col[[cytokine_to_check]].values.astype("float32")
    
    corr_lt = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_raw)):
        print(f"Fold {fold+1}/5")
        
        X_train_raw = X_raw[train_index]
        X_val_raw   = X_raw[val_index]
        y_train     = y[train_index]
        y_val       = y[val_index]
        
        # scale using training fold only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype("float32")
        X_val   = scaler.transform(X_val_raw).astype("float32")
        
        tf.keras.backend.clear_session()
        model = build_fcnn(X_train.shape[1])
        
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=128,      # larger batch = faster
            verbose=0,
            callbacks=[early_stop]
        )
        
        y_pred = model.predict(X_val, verbose=0).ravel()
        y_true = y_val.ravel()
        
        corr, p_value = pearsonr(y_true, y_pred)
        corr_lt.append(corr)
    
    mean_corr = np.mean(corr_lt)
    corr_new_tmp.append(mean_corr)
    
    print("Mean CV Pearson:", mean_corr)

CV_df = pd.DataFrame(
    {"CV": corr_new_tmp},
    index=cyt_tmp
)
# -

CV_df.to_csv("/data/kumarr17/supplemetary_table_cyt_manuscript/CD14_cv.csv")


