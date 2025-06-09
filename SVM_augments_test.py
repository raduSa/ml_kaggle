import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import random

from skimage.feature import hog
from skimage import util as sk_util
from skimage.transform import rotate, AffineTransform, warp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

img_size = (32, 32)

def random_rotation(img, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    return img.rotate(angle)

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def jitter_color(img, b=0.2, c=0.2):
    img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-b, b))
    img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-c, c))
    return img

def affine_transform(img, t=0.1, s=(0.9, 1.1)):
    dx = random.uniform(-t, t) * img.width
    dy = random.uniform(-t, t) * img.height
    sc = random.uniform(*s)

    transform = AffineTransform(translation=(dx, dy), scale=(sc, sc))
    arr = np.array(img)
    warped = warp(arr, transform.inverse, mode='edge', preserve_range=True).astype(np.uint8)
    return Image.fromarray(warped)

def add_noise(img, mode='gaussian'):
    img_np = np.array(img) / 255.0
    noisy = sk_util.random_noise(img_np, mode=mode)
    return Image.fromarray((noisy * 255).astype(np.uint8))

def get_confusion_matrix(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

def load_data_with_augmentation(csv_path, image_dir, augment_funcs=None):
    df = pd.read_csv(csv_path)
    x, y = list(), list()

    for _, row in df.iterrows():
        file_name = row['image_id'].strip() + ".png"
        label = row['label']
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB").resize(img_size)

        # Add augmented imgs
        if augment_funcs:
            for aug in augment_funcs:
                aug_img = aug(img)
                aug_img = aug_img.resize(img_size)
                features = np.array(aug_img).flatten()
                x.append(features)
                y.append(label)

        # Include original img
        img = img.resize(img_size)
        features = np.array(img).flatten()

        x.append(features)
        y.append(label)

    return np.array(x), np.array(y)

#main

augmentations = [
    #random_rotation,
    # horizontal_flip,
    #color_jitter,
    affine_transform,
    #add_noise
]

x_train, y_train = load_data_with_augmentation("train.csv", "train", augmentations)
x_val, y_val = load_data_with_augmentation("validation.csv", "validation")

print(f"Training data")
print(x_train, x_train.shape)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# Train SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(x_train_scaled, y_train)

# Predict and evaluate
y_pred = svm_model.predict(x_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

get_confusion_matrix(y_val, y_pred)