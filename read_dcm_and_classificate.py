import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import joblib
from glob import glob
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_dcm_image(file_path):
    raw_image = tf.io.read_file(file_path)
    img = tfio.image.decode_dicom_image(raw_image, dtype=tf.float32)
    img = img / tf.reduce_max(img)
    img = tf.image.resize(img, (64, 64))
    if len(img.shape) == 2:
        img = tf.expand_dims(img, axis=-1)
    return np.array(img).flatten()

# Verifica se os argumentos foram passados corretamente
if len(sys.argv) < 2:
    print("Uso: python script.py <diretorio_dicom>")
    sys.exit(1)

# O primeiro argumento é o diretório contendo arquivos DICOM
dicom_dir = sys.argv[1]
dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))

if len(dicom_files) == 0:
    print("Nenhum arquivo DICOM encontrado no diretório fornecido.")
    sys.exit(1)

# Exemplo de rótulos binários (ajuste conforme a necessidade)
labels = [0 if i % 2 == 0 else 1 for i in range(len(dicom_files))]

# Pré-processar as imagens DICOM
data = np.array([load_dcm_image(file) for file in dicom_files])

# Dividir os dados em treino e teste usando cuML
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Treinamento do modelo SVM com cuML
if os.path.exists('svm_model.joblib'):
    print('modelo carregado')
    clf = joblib.load('svm_model.joblib')
else:
    print('treinando modelo')
    clf = SVC(kernel='linear')  # Usando um kernel linear para SVM
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'svm_model.joblib')


# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo SVM: {accuracy * 100:.2f}%')

# Exemplo de como classificar um novo arquivo DICOM
new_file = 'A3A791BD.dcm'
new_image = load_dcm_image(new_file)

# Fazer a previsão
prediction = clf.predict([new_image])
print(f'Previsão da nova imagem: {prediction}')
