import os

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.optimizers import Adam
import numpy as np
from PIL import Image
from keras.src.layers import MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


model_3 = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax'),
])

def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        file_name = row[0].strip() + ".png"
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img).flatten()
        X.append(img_array)
    return np.array(X)

def build_model(model):
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load tests
X_test = load_data("test.csv", "test")

X_test = X_test.reshape(len(X_test), 100, 100, 3)

model = build_model(model_3)
model.load_weights('model_3.weights.h5')
predictions = model.predict(X_test)
print(predictions.argmax(axis=1))