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

import pandas as pd
import re
from functools import reduce
from scipy.stats import pearsonr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from scipy.stats import zscore
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
import seaborn as sns
import math

df_tcga=pd.read_csv("/data/kumarr17/merged_sample_mtx/TCGA_Merged_mRNA_Expression.tsv", index_col=0)
df_ligand_rec_pair=pd.read_csv("/data/kumarr17/human_ligand_receptor_pairs.csv")#list(pd.read_csv("/data/kumarr17/DL_model_input/df_cyt_interest.csv",index_col='Unnamed: 0')['cyt_interest']) ## list of cytokine to train model
cyt_of_interest=list(df_ligand_rec_pair['ligand'].unique())
cyt_of_interest=list(set(list(df_tcga.columns))&set(cyt_of_interest))

# +
#cyt_of_interest=list(pd.read_csv("/data/kumarr17/DL_model_input/df_cyt_interest.csv",index_col='Unnamed: 0')['cyt_interest']) ## list of cytokine to train model
## Feature vector
with open("/data/kumarr17/common_features_ver1.txt", "r") as f:
    model_feature_list = [line.strip() for line in f]

filtered_model_feature_list = [x for x in model_feature_list if x not in set(cyt_of_interest)]
# -

### Importing the comparison dataset for single cell gene expression pseudo bulk
cyt_of_interest1=list(pd.read_csv("/data/kumarr17/DL_model_input/df_cyt_interest.csv",index_col='Unnamed: 0')['cyt_interest']) ## list of cytokine to train model
pseudo_bulk_gene_expression_df=pd.read_csv("/data/kumarr17/single_cell_cross_tissue_ML_analysis/pseudo_bulk_gene_expression.csv",index_col='sampleID')
epi_gene_expression_df=pd.read_csv("/data/kumarr17/single_cell_cross_tissue_ML_analysis/Epithelial_gene_expression.csv",index_col='sampleID')
tcga_bulk_gene_expression_df=pd.read_csv("/data/kumarr17/merged_sample_mtx/TCGA_Merged_mRNA_Expression.tsv", index_col='Unnamed: 0')
tcga_epi_gene_expression_df=pd.read_csv("/data/kumarr17/merged_sample_mtx/"+'Cancer'+"_Merged_mRNA_Expression.tsv",index_col='Unnamed: 0')

# +
#cyt_of_interest=list(pd.read_csv("/data/kumarr17/DL_model_input/df_cyt_interest.csv",index_col='Unnamed: 0')['cyt_interest']) ## list of cytokine to train model
## Feature vector
with open("/data/kumarr17/common_features_ver1.txt", "r") as f:
    model_feature_list = [line.strip() for line in f]

filtered_model_feature_list = [x for x in model_feature_list if x not in set(cyt_of_interest1)]
# -

df_tcga_features=tcga_bulk_gene_expression_df[filtered_model_feature_list]
df_tcga_cyt_col=tcga_bulk_gene_expression_df[cyt_of_interest1]

# +
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr


# Features: samples × genes
X = df_tcga_features

# Targets: samples × cytokines
Y = df_tcga_cyt_col


# Same CV as FCNN
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


ridge_results = []
ridge_prediction_df = pd.DataFrame(
    index=X.index,
    columns=Y.columns,
    dtype=float
)


for cytokine in Y.columns:

    print("Running Ridge:", cytokine)

    y = Y[cytokine]

    # Keep samples with available target
    valid = y.notna()

    X_temp = X.loc[valid]
    y_temp = y.loc[valid]


    # IMPORTANT:
    # scaler is fitted ONLY on training fold
    model = make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=1.0
        )
    )


    # Held-out prediction for every sample
    y_pred = cross_val_predict(
        model,
        X_temp,
        y_temp,
        cv=kf,
        n_jobs=-1
    )


    # CV Pearson correlation
    r, p = pearsonr(
        y_temp.values,
        y_pred
    )


    ridge_results.append({
        'cytokine': cytokine,
        'Ridge_CV': r,
        'pvalue': p
    })


    ridge_prediction_df.loc[
        y_temp.index,
        cytokine
    ] = y_pred


ridge_cv_df = pd.DataFrame(ridge_results)

ridge_cv_df.head()
ridge_cv_df.to_csv("/data/kumarr17/ridge_cv_df.csv")
# -


