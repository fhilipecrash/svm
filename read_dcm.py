import pydicom
import argparse
import cv2
import cupy as cp
import numpy as np
import joblib
import sys
import os
from glob import glob
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

def anisotropic_diffusion_with_median_filter_gpu(img, num_iter=5, kappa=50, gamma=0.1):
    img_gpu = cp.array(img, dtype=cp.float32)
    for i in range(num_iter):
        dx = cp.gradient(img_gpu, axis=1)
        dy = cp.gradient(img_gpu, axis=0)
        grad_magnitude = cp.sqrt(dx ** 2 + dy ** 2)
        c = cp.exp(-(grad_magnitude / kappa) ** 2)
        img_gpu += gamma * (c * dx + c * dy)
    img_cpu = cp.asnumpy(img_gpu)
    img_cpu = cv2.GaussianBlur(img_cpu,(5,5),0)
    return img_cpu

def crop_breast_region(img):
    # Converte a imagem para o intervalo [0, 255] e tipo uint8
    image_uint8 = (img * 255).astype(cp.uint8)

    black_pixels = cp.sum(image_uint8 == 0)
    total_pixels = image_uint8.size
    black_ratio = black_pixels / total_pixels

    # Verificar se a maior parte da imagem contém fundo preto
    if black_ratio > 0.5:  # Ajuste este valor se necessário
        image_uint8 = cv2.bitwise_not(image_uint8)

    # Aplica limiar para segmentação
    _, binary_image = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontra contornos e seleciona o maior
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Obtém o retângulo delimitador
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calcula o corte quadrado
    side_length = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    start_x = max(center_x - side_length // 2, 0)
    start_y = max(center_y - side_length // 2, 0)
    end_x = min(center_x + side_length // 2, img.shape[1])
    end_y = min(center_y + side_length // 2, img.shape[0])

    # Realiza o corte
    cropped_image = img[start_y:end_y, start_x:end_x]
    return cropped_image

def load_dcm_image(file_path):
    print(file_path)
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(cp.float32)
    img = (img - cp.min(img)) / (cp.max(img) - cp.min(img) + 1e-7)

    img = anisotropic_diffusion_with_median_filter_gpu(img)
    img = cv2.resize(img, (800, 600))
    img = crop_breast_region(img)

    # if len(img.shape) == 2:
    #     img = np.expand_dims(img, axis=-1)
    cv2.imshow("pydicom.png", img)
    cv2.waitKey(0)
    return img.flatten()

load_dcm_image("B1_F602D90C.dcm")

# parser = argparse.ArgumentParser(description='DICOM classification with SVM, RandomForest, DecisionTree, GradientBoosting, or KNeighbors.')
# parser.add_argument('--train', type=str, help='Path to the DICOM directory for training.')
# parser.add_argument('--dcm', type=str, required=True, help='Path to a DICOM file for classification.')
# parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'random_forest', 'decision_tree', 'gradient_boosting', 'knn'], help='Classifier to use: svm, random_forest, decision_tree, gradient_boosting, or knn')
# parser.add_argument('--cv', type=int, default=5, help='Number of folds for cross-validation.')
# args = parser.parse_args()

# model_file = f"{args.classifier}_model.joblib"

# if args.train is not None:
#     print(f'Training {args.classifier} model with {args.cv}-fold cross-validation...')
#     dicom_dir = args.train
#     dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))

#     if len(dicom_files) == 0:
#         print("No DICOM files found in the given directory.")
#         sys.exit(1)

#     labels = [0 if i % 2 == 0 else 1 for i in range(len(dicom_files))]
#     data = np.array([load_dcm_image(file) for file in dicom_files])

#     if args.classifier == "svm":
#         clf = SVC(kernel='linear')
#     elif args.classifier == "random_forest":
#         clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     elif args.classifier == "decision_tree":
#         clf = DecisionTreeClassifier(random_state=42)
#     elif args.classifier == "gradient_boosting":
#         clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
#     elif args.classifier == "knn":
#         clf = KNeighborsClassifier(n_neighbors=5)

#     cross_val_scores = cross_val_score(clf, data, labels, cv=args.cv)
#     print(f'{args.classifier.capitalize()} model cross-validation accuracy: {cross_val_scores.mean() * 100:.2f}% ± {cross_val_scores.std() * 100:.2f}%')

#     clf.fit(data, labels)
#     model_data = {
#         'model': clf,
#         'cv_accuracy_mean': cross_val_scores.mean(),
#         'cv_accuracy_std': cross_val_scores.std()
#     }
#     joblib.dump(model_data, model_file)
#     print(f'Model trained and saved as {model_file}')
# elif os.path.exists(model_file):
#     print(f'Loaded {args.classifier} model!')
#     model_data = joblib.load(model_file)
#     clf = model_data['model']
#     cv_accuracy_mean = model_data['cv_accuracy_mean']
#     cv_accuracy_std = model_data['cv_accuracy_std']
#     print(f'Cross-validation accuracy of the loaded model: {cv_accuracy_mean * 100:.2f}% ± {cv_accuracy_std * 100:.2f}%')
# else:
#     print(f'No trained model found for {args.classifier}. Use --train to train a new model.')
#     sys.exit(1)

# dcm_file = args.dcm
# test_image = load_dcm_image(dcm_file)
# prediction = clf.predict([test_image])
# print(f'Prediction for {dcm_file}: {prediction}')
