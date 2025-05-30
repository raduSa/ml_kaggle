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

# test1 - 0.64 x 0.9315 -> overfit
model_1 = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax'),
    ])

# test_2 - 0.7058 x 0.6835 / tried dropout
model_2 = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax'),
])

# / added more layers, couldn't overfit
model_3 = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
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
        label = row[1]
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img).flatten()
        X.append(img_array)
        y.append(label)
    return np.array(X), np.array(y)

def build_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Load training and validation data
X_train, Y_train = load_data("train.csv", "train")
X_val, Y_val = load_data("validation.csv", "validation")

X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_val = X_val.reshape(len(X_val), 100, 100, 3)
Y_val = Y_val.reshape(len(Y_val), 1)

# Preprocessing (subtract mean)
mean = X_train.mean(axis=0)
X_train = X_train - mean
X_val = X_val - mean

Y_train = to_categorical(Y_train, num_classes=5)
Y_val = to_categorical(Y_val, num_classes=5)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", Y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", Y_val.shape)

# Prepare data
batch_size = 64

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values
    rotation_range=20,           # Random rotation in degrees
    width_shift_range=0.1,       # Horizontal shift (10%)
    height_shift_range=0.1,      # Vertical shift (10%)
    zoom_range=0.1,              # Random zoom
    horizontal_flip=True,        # Random horizontal flip
    fill_mode='nearest'          # Fill in newly created pixels
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
validation_generator = test_datagen.flow(X_val, Y_val, batch_size=batch_size)

# build model
model = build_model(model_3)

model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping])

model.evaluate(X_val, Y_val)
model.save_weights('first_try.h5')  # always save your weights after training or during training
