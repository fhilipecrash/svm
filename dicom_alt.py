import sys
import os
import numpy as np
import joblib
from glob import glob
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import pydicom
import cv2
import argparse

def load_dcm_image(file_path):
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    if np.max(img) > 0:
        img = img / np.max(img)
    img_resized = cv2.resize(img, (128, 128))
    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)
    return img_resized.flatten()

parser = argparse.ArgumentParser(description='DICOM classification with SVM, RandomForest, or DecisionTree.')
parser.add_argument('--train', type=str, help='Path to the DICOM directory for training.')
parser.add_argument('--dcm', type=str, required=True, help='Path to a DICOM file for classification.')
parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'random_forest', 'decision_tree'], help='Classifier to use: svm, random_forest, or decision_tree')
parser.add_argument('--cv', type=int, default=5, help='Number of folds for cross-validation.')
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
    data = np.array([load_dcm_image(file) for file in dicom_files])

    # Escolhe o classificador baseado na opção fornecida
    if args.classifier == "svm":
        clf = SVC(kernel='linear')
    elif args.classifier == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.classifier == "decision_tree":
        clf = DecisionTreeClassifier(random_state=42)

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
