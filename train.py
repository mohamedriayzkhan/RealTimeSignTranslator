import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==========================
# LOAD DATASET
# ==========================

data = pd.read_csv("dataset/dataset.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print("Dataset Shape:", X.shape)

# ==========================
# ENCODE LABELS
# ==========================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

y_cat = to_categorical(y_encoded)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODEL
# ==========================

model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# SAVE MODEL
# ==========================

model.save("models/sign_model.keras")

loss, acc = model.evaluate(X_test, y_test)

print(f"\nAccuracy: {acc*100:.2f}%")
print("Model saved successfully!")