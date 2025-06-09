import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

img_size = (10, 10)

def get_confusion_matrix(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    x, y = list(), list()
    for _, row in df.iterrows():
        file_name = row['image_id'].strip() + ".png"
        label = row['label']

        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_size)
        img_array = np.array(img).flatten()

        x.append(img_array)
        y.append(label)
    return np.array(x), np.array(y)

# Load data
x_train, y_train = load_data("train.csv", "train")
x_val, y_val = load_data("validation.csv", "validation")
# Check
print(f"Training data")
print(x_train, x_train.shape)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# # Train SVM
# svm_model = SVC(kernel='linear', C=0.1)
# svm_model.fit(X_train_scaled, y_train)
#
# # Predict and eval
# y_pred = svm_model.predict(x_val_scaled)
# accuracy = accuracy_score(y_val, y_pred)
# print(f"Validation Accuracy: {accuracy:.4f}")
#
# get_confusion_matrix(y_val, y_pred)

# Param search
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

svm = SVC()

grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(x_train_scaled, y_train)

# Report best results
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Evaluate on validation set
best_svm = grid_search.best_estimator_
# Predict and eval
y_pred = best_svm.predict(x_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

get_confusion_matrix(y_val, y_pred)