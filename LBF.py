import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

radius = 3
n_points = 255
method = 'uniform'

def get_lbp_histogram(image):
    lbp = local_binary_pattern(image, n_points, radius, method)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    x, y = list(), list()
    for _, row in df.iterrows():
        file_name = row['image_id'].strip() + ".png"
        label = row['label']

        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("L") # Greyscale
        lbp_hist = get_lbp_histogram(np.array(img))

        x.append(lbp_hist)
        y.append(label)
    return np.array(x), np.array(y)

def get_confusion_matrix(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


# Load data
x_train_hist, y_train = load_data("train.csv", "train")
x_val_hist, y_val = load_data("validation.csv", "validation")

# Standardize features

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_hist)
x_val_scaled = scaler.transform(x_val_hist)

# Plot 2D PCA projection of training data
pca = PCA(n_components=2)
x_train_reduced = pca.fit_transform(x_train_scaled)

plt.figure(figsize=(6, 6))
for label in set(y_train):
    idx = [i for i, l in enumerate(y_train) if l == label]
    plt.scatter(x_train_reduced[idx, 0], x_train_reduced[idx, 1], label=f"Class {label}", s=10)
plt.title("PCA of LBP Histograms (Train Set)")
plt.legend()
plt.show()


print(f"Training data:")
print(x_train_scaled, x_train_scaled.shape)

# Train SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(x_train_scaled, y_train)

# Predict and evaluate
y_pred = svm_model.predict(x_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Compute confusion matrix
get_confusion_matrix(y_val, y_pred)