import sys
import os
import numpy as np
import cupy as cp
import joblib
from glob import glob
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import cv2
import argparse

def anisotropic_diffusion_with_median_filter_gpu(img, num_iter=5, kappa=50, gamma=0.1):
    img_gpu = cp.array(img, dtype=cp.float32)
    for i in range(num_iter):
        dx = cp.gradient(img_gpu, axis=1)
        dy = cp.gradient(img_gpu, axis=0)
        grad_magnitude = cp.sqrt(dx ** 2 + dy ** 2)
        c = cp.exp(-(grad_magnitude / kappa) ** 2)
        img_gpu += gamma * (c * dx + c * dy)
    img_cpu = cp.asnumpy(img_gpu)
    img_cpu = cv2.medianBlur(img_cpu.astype(np.float32), 3)
    return img_cpu

def crop_breast_region(image):
    image_uint8 = (image * 255).astype(np.uint8)

    black_pixels = np.sum(image_uint8 == 0)
    total_pixels = image_uint8.size
    black_ratio = black_pixels / total_pixels

    if black_ratio > 0.5:
        image_uint8 = cv2.bitwise_not(image_uint8)

    _, binary_image = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    side_length = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    start_x = max(center_x - side_length // 2, 0)
    start_y = max(center_y - side_length // 2, 0)
    end_x = min(center_x + side_length // 2, image.shape[1])
    end_y = min(center_y + side_length // 2, image.shape[0])

    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

def load_png_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

    img_cropped = crop_breast_region(img)
    img_processed = anisotropic_diffusion_with_median_filter_gpu(img_cropped)

    window_center, window_width = 40, 80
    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)
    img_clipped = np.clip(img_processed, lower_bound, upper_bound)
    img_normalized = (img_clipped - lower_bound) / (upper_bound - lower_bound + 1e-7)
    img_resized = cv2.resize(img_normalized, (256, 256))

    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)
    return img_resized.flatten()

parser = argparse.ArgumentParser(description='PNG classification with SVM, RandomForest, DecisionTree, GradientBoosting, or KNeighbors.')
parser.add_argument('--train', type=str, help='Path to the PNG directory for training.')
parser.add_argument('--png', type=str, required=True, help='Path to a PNG file for classification.')
parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'random_forest', 'decision_tree', 'gradient_boosting', 'knn'], help='Classifier to use: svm, random_forest, decision_tree, gradient_boosting, or knn')
parser.add_argument('--cv', type=int, default=5, help='Number of folds for cross-validation.')
args = parser.parse_args()

model_file = f"{args.classifier}_model.joblib"

if args.train is not None:
    print(f'Training {args.classifier} model with {args.cv}-fold cross-validation...')
    png_dir = args.train
    png_files = glob(os.path.join(png_dir, "*.png"))

    if len(png_files) == 0:
        print("No PNG files found in the given directory.")
        sys.exit(1)

    labels = [0 if i % 2 == 0 else 1 for i in range(len(png_files))]
    data = np.array([load_png_image(file) for file in png_files])

    if args.classifier == "svm":
        clf = SVC(kernel='linear')
    elif args.classifier == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.classifier == "decision_tree":
        clf = DecisionTreeClassifier(random_state=42)
    elif args.classifier == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif args.classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)

    cross_val_scores = cross_val_score(clf, data, labels, cv=args.cv)
    print(f'{args.classifier.capitalize()} model cross-validation accuracy: {cross_val_scores.mean() * 100:.2f}% ± {cross_val_scores.std() * 100:.2f}%')

    clf.fit(data, labels)
    model_data = {
        'model': clf,
        'cv_accuracy_mean': cross_val_scores.mean(),
        'cv_accuracy_std': cross_val_scores.std()
    }
    joblib.dump(model_data, model_file)
    print(f'Model trained and saved as {model_file}')
elif os.path.exists(model_file):
    print(f'Loaded {args.classifier} model!')
    model_data = joblib.load(model_file)
    clf = model_data['model']
    cv_accuracy_mean = model_data['cv_accuracy_mean']
    cv_accuracy_std = model_data['cv_accuracy_std']
    print(f'Cross-validation accuracy of the loaded model: {cv_accuracy_mean * 100:.2f}% ± {cv_accuracy_std * 100:.2f}%')
else:
    print(f'No trained model found for {args.classifier}. Use --train to train a new model.')
    sys.exit(1)

png_file = args.png
new_image = load_png_image(png_file)
prediction = clf.predict([new_image])
print(f'Prediction for {png_file}: {prediction}')
