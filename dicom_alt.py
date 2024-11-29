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
from sklearn.model_selection import cross_val_score, train_test_split
import pydicom
import cv2
import argparse
import multiprocessing

def anisotropic_diffusion_with_median_filter_gpu(img, num_iter=5, kappa=50, gamma=0.1):
    # Transfere a imagem para a GPU
    img_gpu = cp.array(img, dtype=cp.float32)

    # Realiza a difusão anisotrópica
    for i in range(num_iter):
        dx = cp.gradient(img_gpu, axis=1)
        dy = cp.gradient(img_gpu, axis=0)
        grad_magnitude = cp.sqrt(dx ** 2 + dy ** 2)
        c = cp.exp(-(grad_magnitude / kappa) ** 2)
        img_gpu += gamma * (c * dx + c * dy)
    
    # Converte a imagem de volta para CPU para aplicar o filtro mediano
    img_cpu = cp.asnumpy(img_gpu)

    # Aplica o filtro mediano com OpenCV usando suporte a CUDA
    # img_cpu = cv2.medianBlur(img_cpu.astype(np.float32), 3)
    img_cpu = cv2.medianBlur(img_cpu.astype(np.float32), 3)

    return img_cpu

def load_dcm_image(file_path, lol=0):
    print(lol)
    dcm_data = pydicom.dcmread(file_path)

    img = dcm_data.pixel_array.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

    # Aplica difusão anisotrópica e filtro mediano
    img = anisotropic_diffusion_with_median_filter_gpu(img)

    window_center, window_width = 40, 80
    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)
    img = np.clip(img, lower_bound, upper_bound)
    img = (img - lower_bound) / (upper_bound - lower_bound + 1e-7)
    img_resized = cv2.resize(img, (256, 256))

    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)
    return img_resized.flatten()

def process_dicom_files_in_parallel(dicom_files, labels):
    """
    Processa múltiplos arquivos DICOM em paralelo.
    """
    # Adiciona o índice para debug ou outros usos
    files_with_indices = [(file, idx + 1) for idx, file in enumerate(dicom_files)]
    print(files_with_indices)
    
    # Cria um pool de processos
    with multiprocessing.Pool(processes=args.proc) as pool:
        # Processa os arquivos DICOM em paralelo
        processed_data = pool.starmap(load_dcm_image, files_with_indices)
    
    return np.array(processed_data)


parser = argparse.ArgumentParser(description='DICOM classification with SVM, RandomForest, DecisionTree, GradientBoosting, or KNeighbors.')
parser.add_argument('--train', type=str, help='Path to the DICOM directory for training.')
parser.add_argument('--dcm', type=str, required=True, help='Path to a DICOM file for classification.')
parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'random_forest', 'decision_tree', 'gradient_boosting', 'knn'], help='Classifier to use: svm, random_forest, decision_tree, gradient_boosting, or knn')
parser.add_argument('--cv', type=int, default=5, help='Number of folds for cross-validation.')
parser.add_argument('--proc', type=int, default=4, help='Number of processes for parallel processing.')
args = parser.parse_args()

# Define o nome do arquivo do modelo com base no classificador escolhido
model_file = f"{args.classifier}_model.joblib"

if args.train is not None:
    print(f'Training {args.classifier} model with {args.cv}-fold cross-validation...')
    dicom_dir = args.train
    dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))

    if len(dicom_files) == 0:
        print("No DICOM files found in the given directory.")
        sys.exit(1)

    labels = [0 if i % 2 == 0 else 1 for i in range(len(dicom_files))]

    # Processa os arquivos DICOM em paralelo
    data = process_dicom_files_in_parallel(dicom_files, labels)

    # Escolhe o classificador baseado na opção fornecida
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

    # Validação cruzada
    cross_val_scores = cross_val_score(clf, data, labels, cv=args.cv)
    print(f'{args.classifier.capitalize()} model cross-validation accuracy: {cross_val_scores.mean() * 100:.2f}% ± {cross_val_scores.std() * 100:.2f}%')

    # Treinamento final em todos os dados de treino para salvar o modelo
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

# Classificação de um novo arquivo DICOM
dcm_file = args.dcm
new_image = load_dcm_image(dcm_file)
prediction = clf.predict([new_image])
print(f'Prediction for {dcm_file}: {prediction}')
