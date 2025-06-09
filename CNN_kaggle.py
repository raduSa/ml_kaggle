import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.src.layers import Flatten, GlobalMaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --------------------------------------
#               Load data
# --------------------------------------

def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    x, y = list(), list()
    for _, row in df.iterrows():
        file_name = row['image_id'].strip() + ".png"
        label = row['label']
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        x.append(img_array)
        y.append(label)
    return np.array(x), np.array(y)


X_train_org, Y_train = load_data("train.csv", "train")
X_val_org, Y_val = load_data("validation.csv", "validation")

# One-hot encode labels
Y_train = to_categorical(Y_train, num_classes=5)
Y_val = to_categorical(Y_val, num_classes=5)


# -------- Load data with class based mean subtraction --------
from collections import defaultdict

# Function for loading training data (also returns training data mean)
def load_train_data_with_class_mean(csv_path, image_dir, image_size=(100, 100)):
    df = pd.read_csv(csv_path)
    x, y = list(), list()
    class_images = defaultdict(list)

    for _, row in df.iterrows():
        file_name = row['image_id'].strip() + ".png"
        label = int(row['label'])
        img_path = os.path.join(image_dir, file_name)

        img = Image.open(img_path).convert("RGB").resize(image_size)
        img_array = np.array(img) / 255.0

        class_images[label].append(img_array)
        x.append(img_array)
        y.append(label)

    X = np.array(x)
    y = np.array(y)

    # Get mean per class
    class_means = {label: np.mean(imgs, axis=0) for label, imgs in class_images.items()}

    # Subtract class-wise mean
    X_centered = np.array([img - class_means[label] for img, label in zip(X, y)])

    # Also compute train mean used for val/test
    train_mean = np.mean(X, axis=0)

    return X_centered, y, train_mean

# Function for loading val/test data
def load_data_with_global_mean(csv_path, image_dir, global_mean, image_size=(100, 100)):
    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        file_name = row[0].strip() + ".png"
        label = int(row[1])
        img_path = os.path.join(image_dir, file_name)

        img = Image.open(img_path).convert("RGB").resize(image_size)
        img_array = np.array(img) / 255.0
        X.append(img_array)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_centered = X - global_mean
    return X_centered, y

'''X_train, Y_train, global_mean = load_train_data_with_class_mean("/kaggle/input/image-deep-fake/train.csv", "/kaggle/input/image-deep-fake/train")
X_val, Y_val = load_data_with_global_mean("/kaggle/input/image-deep-fake/validation.csv", "/kaggle/input/image-deep-fake/validation", global_mean)
#X_test, Y_test = load_data_with_global_mean("test.csv", "test", global_mean)

# One-hot encode labels
Y_train = to_categorical(Y_train, num_classes=5)
Y_val = to_categorical(Y_val, num_classes=5)'''

# --------------------------------------
#             Preprocessing
# --------------------------------------

# Normalize using mean subtraction
mean_pixel = X_train_org.mean(axis=(0, 1, 2), keepdims=True)
X_train = (X_train_org - mean_pixel) / 255.0
X_val = (X_val_org - mean_pixel) / 255.0

'''X_train = X_train_org / 255.0
X_val = X_val_org / 255.0'''

'''X_train = X_train_org
X_val = X_val_org'''

print("Training data")
print(X_train.shape)
print(X_val.shape)
print("Training labels")
print(Y_train.shape)
print(Y_val.shape)

# --------------------------------------
#           Data generators
# --------------------------------------

# -------- No augment --------
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

'''train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
val_generator = test_datagen.flow(X_val, Y_val, batch_size=batch_size)'''

# -------- With augmentations --------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    #horizontal_flip=True,
    fill_mode='nearest'
)

# For validation, we only rescale
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# I used this one
train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
val_generator = test_datagen.flow(X_val, Y_val, batch_size=batch_size)

# -------- Per image standardization --------
from tensorflow.keras.layers import Layer
class PerImageStandardization(Layer):
    def __init__(self, **kwargs):
        super(PerImageStandardization, self).__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
        return (inputs - mean) / (std + 1e-7)

# -------- Using batch mix-up --------
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class MixupGenerator(Sequence):
    def __init__(self, x, y, batch_size=32, alpha=0.4, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]

        # Mixup
        lam = np.random.beta(self.alpha, self.alpha)
        index_array = np.random.permutation(len(x_batch))
        x1, x2 = x_batch, x_batch[index_array]
        y1, y2 = y_batch, y_batch[index_array]
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2

        return x_mix, y_mix


# --------------------------------------
#              Build model
# --------------------------------------

# List of all model architectures tried
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

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax'),
])

# added more layers, tried batch normalization layer
model_3 = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(100, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax'),
])

# tired adding regularization
model_5 = Sequential([
    Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), padding='same', activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(100, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(5, activation='softmax'),
])

def build_cnn():
    model = model_5
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------------------
#              Train model
# --------------------------------------
# Use early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
# Used the mixup gen for best submission
mixup_gen = MixupGenerator(X_train, Y_train, batch_size=64, alpha=0.4)

model = build_cnn()

model.fit(
    mixup_gen,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# -------------------------------
#       Evaluation & Save
# -------------------------------

model.evaluate(X_val, Y_val)
model.save_weights('trial_8.2.weights.h5')

# -------------------------------
#       Create Submission
# -------------------------------

test_df = pd.read_csv('test.csv')
image_dir = 'test'

def load_test_images(test_df, image_dir):
    X_test = []
    for fname in test_df['image_id']:
        image_path = os.path.join(image_dir, fname.strip() + '.png')
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        X_test.append(img_array)
    X_test = np.array(X_test)
    return X_test

X_test = load_test_images(test_df, image_dir)
X_test = (X_test - mean_pixel) / 255.0

# -------- Predict --------
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# -------- Generate submission.csv --------
submission_df = pd.DataFrame({
    'image_id': test_df['image_id'],
    'label': predicted_labels
})
submission_df.to_csv('new_submission.csv', index=False)

# -------------------------------
#      Show Confusion matrix
# -------------------------------

# Get true labels and predictions
y_true = np.argmax(Y_val, axis=1)
y_pred = np.argmax(model.predict(X_val), axis=1)

# Compute and display confusion matrix
cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
#      Hyperparam search
# -------------------------------

from tensorflow.keras.regularizers import l2
import keras_tuner as kt

# Decide what type of pooling layer to use
def get_pooling_layer(pool_type, global_pool=False):
    if global_pool:
        return GlobalAveragePooling2D() if pool_type == 'avg' else GlobalMaxPooling2D()
    return AveragePooling2D(pool_size=(2, 2)) if pool_type == 'avg' else MaxPooling2D(pool_size=(2, 2))

def get_model(hp):
    pool_type = hp.Choice('pooling_type', ['max', 'avg'])
    l2_factor = hp.Choice('l2_regularization', [0.0, 1e-4, 1e-3, 1e-2])

    model = Sequential()

    # Layer 1
    model.add(Conv2D(
        filters=hp.Choice('filters_1', [32, 64, 128]),
        kernel_size=hp.Choice('kernel_size_1', [3, 5, 7]),
        activation='relu',
        input_shape=(100, 100, 3),
        padding='same',
        kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))

    # Layer 2
    model.add(Conv2D(
        filters=hp.Choice('filters_2', [64, 128, 256]),
        kernel_size=hp.Choice('kernel_size_2', [3, 5]),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(get_pooling_layer(pool_type))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

    # Layer 3
    model.add(Conv2D(
        filters=hp.Choice('filters_3', [64, 128, 256]),
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(get_pooling_layer(pool_type))
    model.add(Dropout(hp.Float('dropout_3', 0.2, 0.5, step=0.1)))

    # Global Pooling
    global_pool_type = hp.Choice('global_pool_type', ['avg', 'max'])
    model.add(get_pooling_layer(global_pool_type, global_pool=True))

    # Dense Layers
    model.add(Dense(
        hp.Int('dense_units', 64, 256, step=64),
        activation='relu',
        kernel_regularizer=l2(l2_factor)))
    model.add(Dropout(hp.Float('dropout_dense', 0.3, 0.6, step=0.1)))
    model.add(Dense(5, activation='softmax'))

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

from keras_tuner import RandomSearch

tuner = RandomSearch(
    get_model,
    objective='val_accuracy',
    max_trials=53,
    directory='tuning_dir',
    project_name='animal_classifier'
)

best_model = tuner.get_best_models(num_models=1)[0]

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)