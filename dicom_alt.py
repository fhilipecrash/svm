import sys
import os
import numpy as np
import joblib
from glob import glob
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pydicom
import cv2
import argparse

def load_dcm_image(file_path):
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    if np.max(img) > 0:
        img = img / np.max(img)
    else:
        img = img

    img_resized = cv2.resize(img, (64, 64))
    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)
    
    return img_resized.flatten()

parser = argparse.ArgumentParser(description='SVM training and DICOM classification.')
parser.add_argument('--train', type=str, help='Path to the DICOM directory for training.')
parser.add_argument('--dcm', type=str, required=True, help='Path to a DICOM file for classification.')
args = parser.parse_args()

train_new_model = args.train is not None

if train_new_model:
    print('Training model...')
    dicom_dir = args.train
    dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))

    if len(dicom_files) == 0:
        print("No DICOM files found in the given directory.")
        sys.exit(1)

    labels = [0 if i % 2 == 0 else 1 for i in range(len(dicom_files))]
    data = np.array([load_dcm_image(file) for file in dicom_files])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM model accuracy: {accuracy * 100:.2f}%')

    model_data = {
        'model': clf,
        'accuracy': accuracy
    }
    joblib.dump(model_data, 'svm_model.joblib')
    print('Model trained and saved as svm_model.joblib')
elif os.path.exists('svm_model.joblib'):
    print('Loaded model!')
    model_data = joblib.load('svm_model.joblib')
    clf = model_data['model']
    accuracy = model_data['accuracy']
    print(f'Accuracy of the loaded model: {accuracy * 100:.2f}%')
else:
    print('No trained model found. Use --train to train a new model.')
    sys.exit(1)

dcm_file = args.dcm
new_image = load_dcm_image(dcm_file)

prediction = clf.predict([new_image])
print(f'Image preview {dcm_file}: {prediction}')
