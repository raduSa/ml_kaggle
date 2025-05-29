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

# --- Config ---
IMAGE_SIZE = (32, 32)  # Increased size for better HOG features


# === IMAGE AUGMENTATION FUNCTIONS === #

def random_rotation(img, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    return img.rotate(angle)

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def color_jitter(img, brightness=0.2, contrast=0.2):
    enhancer_b = ImageEnhance.Brightness(img)
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_b.enhance(1 + random.uniform(-brightness, brightness))
    img = enhancer_c.enhance(1 + random.uniform(-contrast, contrast))
    return img

def affine_transform(img, translate_frac=0.1, scale_range=(0.9, 1.1)):
    tx = random.uniform(-translate_frac, translate_frac) * img.width
    ty = random.uniform(-translate_frac, translate_frac) * img.height
    scale = random.uniform(*scale_range)

    transform = AffineTransform(translation=(tx, ty), scale=(scale, scale))
    img_np = np.array(img)
    warped = warp(img_np, transform.inverse, mode='edge', preserve_range=True).astype(np.uint8)
    return Image.fromarray(warped)

def add_noise(img, mode='gaussian'):
    img_np = np.array(img) / 255.0
    noisy = sk_util.random_noise(img_np, mode=mode)
    return Image.fromarray((noisy * 255).astype(np.uint8))


# === HOG FEATURE EXTRACTOR === #

def extract_hog_features_from_pil(features):
    return hog(features, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)


# === DATA LOADER WITH AUGMENTATION === #

def load_data_with_augmentation(csv_path, image_dir, augment_funcs=None, hog=False):
    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        file_name = row[0].strip() + ".png"
        label = row[1]
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)

        if augment_funcs:
            AUGMENT_REPEATS = 1  # number of times to apply each augmentation randomly

            for aug in augment_funcs:
                for _ in range(AUGMENT_REPEATS):
                    aug_img = aug(img)
                    aug_img = aug_img.resize(IMAGE_SIZE)
                    features = np.array(aug_img).flatten()
                    if hog:
                        features = extract_hog_features_from_pil(aug_img)
                    X.append(features)
                    y.append(label)

        # Also include original
        img = img.resize(IMAGE_SIZE)
        features = np.array(img).flatten()
        if hog:
            features = extract_hog_features_from_pil(features)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

#main

augmentations = [
    random_rotation,
    #horizontal_flip,
    #color_jitter,
    affine_transform,
    #add_noise
]

X_train, y_train = load_data_with_augmentation("train.csv", "train", augment_funcs=augmentations)
X_val, y_val = load_data_with_augmentation("validation.csv", "validation")  # No augmentation for validation

print(f"Training data")
print(X_train, X_train.shape)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")