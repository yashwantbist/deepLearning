import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
filepath = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv"
concrete_data = pd.read_csv(filepath)

# Split predictors and target
X = concrete_data.drop(columns=["Strength"])
y = concrete_data["Strength"]

# Normalize predictors
X_norm = (X - X.mean()) / X.std()

# Convert to numpy float32 (recommended)
X_norm = X_norm.to_numpy(dtype=np.float32)
y = y.to_numpy(dtype=np.float32)

n_cols = X_norm.shape[1]

# Build model
def regression_model():
    model = keras.Sequential([
        layers.Input(shape=(n_cols,)),
        layers.Dense(50, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = regression_model()

history = model.fit(
    X_norm, y,
    validation_split=0.3,
    epochs=100,
    batch_size=32,
    verbose=2
)

print("Done. Final val_loss:", history.history["val_loss"][-1])
