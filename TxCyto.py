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
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
# ============================================================
# PATHS
# ============================================================
import os
TXCYTO_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Users should place the downloaded TxCyto data here:
# TxCyto/data/

DATA_DIR = os.path.join(
    TXCYTO_DIR,
    "data"
)

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "common_features_ver3.txt"
)

TRAINING_FILE = os.path.join(
    DATA_DIR,
    "TCGA_Merged_mRNA_Expression.tsv"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"\nRequired TxCyto feature file not found:\n"
        f"{FEATURE_FILE}\n\n"
        f"Please download the TxCyto data and place "
        f"'common_features_ver3.txt' inside the 'data' folder."
    )

if not os.path.exists(TRAINING_FILE):
    raise FileNotFoundError(
        f"\nRequired TxCyto training file not found:\n"
        f"{TRAINING_FILE}\n\n"
        f"Please download the TxCyto data and place "
        f"'TCGA_Merged_mRNA_Expression.tsv' inside the 'data' folder."
    )

#import os

#TXCYTO_DIR = os.path.dirname(
#    os.path.abspath(__file__)
#)

#FEATURE_FILE = os.path.join(
#    TXCYTO_DIR,
#    "common_features_ver3.txt"
#)

#TRAINING_FILE = os.path.join(
#    TXCYTO_DIR,
#    "TCGA_Merged_mRNA_Expression.tsv"
#)

#TCGA_PATH = "/data/kumarr17/merged_sample_mtx/TCGA_Merged_mRNA_Expression.tsv"

#FEATURE_PATH = "/data/kumarr17/common_features_ver3.txt"


# ============================================================
# LOAD TRAINING DATA + FIXED FEATURES
# ============================================================

tcga_features_list = list(
    pd.read_csv(FEATURE_FILE, header=None)[0]
)

df_tcga = pd.read_csv(
    TRAINING_FILE,
    index_col=0
)


# ============================================================
# FCNN MODEL
# ============================================================

def build_fcnn(input_dim):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


# ============================================================
# TXCYTO PREDICTION FUNCTION
# ============================================================

def predict_txcyto(
    test_df,
    cytokines,
    epochs=50,
    batch_size=128,
    verbose=0
):
# ============================================================
# REPRODUCIBILITY
# ============================================================
    tf.keras.backend.clear_session()
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --------------------------------------------------------
    # Cytokine input
    # --------------------------------------------------------

    if isinstance(cytokines, str):
        cytokines = [cytokines]

    cytokines = list(cytokines)
    
    # --------------------------------------------------------
    # Check requested cytokines
    # --------------------------------------------------------
    
    available_cytokines = [
        c for c in cytokines
        if c in df_tcga.columns
    ]
    
    missing_cytokines = [
        c for c in cytokines
        if c not in df_tcga.columns
    ]
    
    
    # Report unavailable cytokines
    if missing_cytokines:
        print(
            "\nWarning: The following cytokines are not available "
            "in TxCyto and will be skipped:"
        )
        print(missing_cytokines)
    
    
    # Stop only if NONE of the requested cytokines are available
    if len(available_cytokines) == 0:
        raise ValueError(
            "None of the requested cytokines are available in TxCyto."
        )
    
    
    print(
        f"\nPredicting {len(available_cytokines)} "
        f"of {len(cytokines)} requested cytokines:"
    )
    
    print(available_cytokines)
    
    
    # Use only available cytokines from this point onward
    cytokines = available_cytokines    
    # --------------------------------------------------------
    # Check test features
    # --------------------------------------------------------

    missing_features = [
        gene for gene in tcga_features_list
        if gene not in test_df.columns
    ]

    if len(missing_features) > 0:

        raise ValueError(
            f"{len(missing_features)} required TxCyto features "
            "are missing from the test dataset.\n"
            f"First missing genes: {missing_features[:20]}"
        )


    # --------------------------------------------------------
    # Select features in EXACT training order
    # --------------------------------------------------------

    df_train_features = df_tcga[tcga_features_list].copy()

    df_test_features = test_df[tcga_features_list].copy()


    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    X_train_raw = (
        df_train_features
        .values
        .astype("float32")
    )

    X_test_raw = (
        df_test_features
        .values
        .astype("float32")
    )


    # --------------------------------------------------------
    # Row-wise z-score
    #
    # IMPORTANT:
    # Each sample is normalized across genes independently.
    # --------------------------------------------------------

    X_train_raw = (
        X_train_raw -
        X_train_raw.mean(axis=1, keepdims=True)
    ) / (
        X_train_raw.std(axis=1, keepdims=True)
        + 1e-8
    )


    X_test_raw = (
        X_test_raw -
        X_test_raw.mean(axis=1, keepdims=True)
    ) / (
        X_test_raw.std(axis=1, keepdims=True)
        + 1e-8
    )

    # --------------------------------------------------------
    # StandardScaler
    #
    # Fit ONLY on TCGA training samples.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train_raw
    ).astype("float32")

    X_test = scaler.transform(
        X_test_raw
    ).astype("float32")


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )


    # --------------------------------------------------------
    # Train cytokine-specific models
    # --------------------------------------------------------

    pred_df_list = []


    for cytokine in cytokines:

        print("\n======================================")
        print(f"Training TxCyto model: {cytokine}")
        print("======================================")


        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        y_train = (
            df_tcga[[cytokine]]
            .values
            .astype("float32")
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = build_fcnn(
            X_train.shape[1]
        )


        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop],
            shuffle=True
        )


        # ----------------------------------------------------
        # Predict cytokine activity
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test,
            verbose=0
        ).ravel()


        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        tmp = pd.DataFrame({

            "sample":
                df_test_features.index,

            "cytokine":
                cytokine,

            "predicted_activity":
                y_pred

        })


        pred_df_list.append(tmp)


    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    pred_long = pd.concat(
        pred_df_list,
        axis=0,
        ignore_index=True
    )


    # --------------------------------------------------------
    # Sample × cytokine matrix
    # --------------------------------------------------------

    pred_matrix = pred_long.pivot(
        index="sample",
        columns="cytokine",
        values="predicted_activity"
    )


    print("\nTxCyto prediction completed.")
    print(f"Samples: {pred_matrix.shape[0]}")
    print(f"Cytokines: {pred_matrix.shape[1]}")
    print(f"Features used: {len(tcga_features_list)}")


    return pred_long, pred_matrix
