import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Parameters
IMAGE_SIZE = (32, 32)  # Keep small for efficiency

def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        file_name = row[0].strip()
        label = row[1]
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB") + ".png"
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img).flatten()
        X.append(img_array)
        y.append(label)
    return np.array(X), np.array(y)

# Load training and validation data
X_train, y_train = load_data("train.csv", "train")
X_val, y_val = load_data("validation.csv", "validation")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define SVM and hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

svm = SVC()

# Grid Search with 3-fold cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Report best results
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Evaluate on validation set
best_svm = grid_search.best_estimator_
val_accuracy = best_svm.score(X_val_scaled, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")
