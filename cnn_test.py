import pydicom
import argparse
import cv2
import cupy as cp
import numpy as np
import joblib
import sys
import os
from glob import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Funções existentes para pré-processamento de imagens
def anisotropic_diffusion_with_median_filter_gpu(img, num_iter=5, kappa=50, gamma=0.1):
    img_gpu = cp.array(img, dtype=cp.float32)
    for i in range(num_iter):
        dx = cp.gradient(img_gpu, axis=1)
        dy = cp.gradient(img_gpu, axis=0)
        grad_magnitude = cp.sqrt(dx ** 2 + dy ** 2)
        c = cp.exp(-(grad_magnitude / kappa) ** 2)
        img_gpu += gamma * (c * dx + c * dy)
    img_cpu = cp.asnumpy(img_gpu)
    img_cpu = cv2.blur(img_cpu, (10, 10))
    return img_cpu

def crop_breast_region(img, photometric_interpretation):
    image_uint8 = (img * 255).astype(cp.uint8)
    if photometric_interpretation == "MONOCHROME2":
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
    end_x = min(center_x + side_length // 2, img.shape[1])
    end_y = min(center_y + side_length // 2, img.shape[0])
    cropped_image = img[start_y:end_y, start_x:end_x]
    return cropped_image

def load_dcm_image(file_path):
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    img = anisotropic_diffusion_with_median_filter_gpu(img)
    img = crop_breast_region(img, dcm_data[0x28, 0x04].value)
    img = cv2.resize(img, (256, 256))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    return img

# Parser para argumentos
parser = argparse.ArgumentParser(description='Classificação de mamografias com CNN.')
parser.add_argument('--train', type=str, help='Diretório de DICOMs para treinamento.')
parser.add_argument('--dcm', type=str, help='Arquivo DICOM para classificação.')
parser.add_argument('--model', type=str, default='cnn_model.h5', help='Arquivo do modelo CNN.')
args = parser.parse_args()

# Caminho do modelo
model_file = args.model

if args.train:
    # Carregar dados de treinamento
    dicom_dir = args.train
    dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))
    if len(dicom_files) == 0:
        print("Nenhum arquivo DICOM encontrado no diretório fornecido.")
        sys.exit(1)

    labels = [0 if i % 2 == 0 else 1 for i in range(len(dicom_files))]  # Labels fictícios para exemplo
    data = np.array([load_dcm_image(file) for file in dicom_files])
    labels = to_categorical(labels, num_classes=2)

    # Divisão dos dados
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Definição do modelo CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinamento
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

    # Salvar modelo
    model.save(model_file)
    print(f"Modelo treinado e salvo como {model_file}")

elif args.dcm:
    if not os.path.exists(model_file):
        print(f"Modelo {model_file} não encontrado. Treine o modelo primeiro usando --train.")
        sys.exit(1)

    # Carregar modelo
    model = load_model(model_file)
    print(f"Modelo {model_file} carregado.")

    # Carregar e preprocessar a imagem DICOM
    dcm_file = args.dcm
    test_image = load_dcm_image(dcm_file)
    test_image = np.expand_dims(test_image, axis=0)  # Adicionar dimensão do batch

    # Predição
    prediction = model.predict(test_image)
    class_idx = np.argmax(prediction)
    print(f"Predição para {dcm_file}: Classe {class_idx} com probabilidade {prediction[0][class_idx]:.2f}")

else:
    print("Forneça --train para treinar ou --dcm para classificar um arquivo DICOM.")
