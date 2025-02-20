import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

def build_model():
    model = Sequential([
        Input((256, 256, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Modelo CNN construído.")
    return model

def create_dataset(images, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def load_data(data_dir):
    image_files = glob(os.path.join(data_dir, "*.png"))
    if not image_files:
        print(f"Nenhuma imagem PNG encontrada no diretório: {data_dir}")
        sys.exit(1)
    labels = [int(file.split("/")[-1].split("_")[0].replace("B", "")) for file in image_files]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    images = np.array([img / 255.0 for img in images]).reshape(-1, 256, 256, 1)
    labels = to_categorical(labels, num_classes=6)
    return images, labels

def main():
    parser = argparse.ArgumentParser(description="Treinar e testar modelo CNN.")
    parser.add_argument("--train", type=str, help="Diretório contendo imagens PNG para treinamento.")
    parser.add_argument("--dcm", type=str, help="Diretório contendo imagens PNG para teste.")
    parser.add_argument("--model", type=str, default="cnn_model.keras", help="Arquivo do modelo CNN.")
    args = parser.parse_args()

    if args.train:
        train_dir = args.train
        model_path = args.model
        BATCH_SIZE = 16

        print("Carregando dados de treinamento...")
        images, labels = load_data(train_dir)
        train_dataset = create_dataset(images, labels, BATCH_SIZE)

        model = build_model()
        
        model.fit(train_dataset, epochs=10, verbose=2)
        
        val_loss, val_acc = model.evaluate(train_dataset, verbose=2)
        print(f"Acurácia na validação: {val_acc:.2%}")
        
        model.save(model_path)
        print(f"Modelo salvo em: {model_path}")

    elif args.dcm:
        test_dir = args.dcm
        model_path = args.model

        if not os.path.exists(model_path):
            print(f"Modelo {model_path} não encontrado. Treine o modelo primeiro usando --train.")
            sys.exit(1)

        print("Carregando modelo para teste...")
        model = load_model(model_path)
        print(f"Modelo {model_path} carregado.")

        print("Carregando dados de teste...")
        test_images, test_labels = load_data(test_dir)

        print("Avaliando o modelo nos dados de teste...")
        predictions = model.predict(test_images, verbose=2)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(test_labels, axis=1)

        accuracy = accuracy_score(y_true, y_pred)

        print("Matriz de Confusão:")
        print(confusion_matrix(y_true, y_pred))
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], target_names=["BI-RADS 0", "BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]))

        print(f"Acurácia nos dados de teste: {accuracy:.2%}")

    else:
        print("Use --train para treinamento ou --dcm para teste.")
        sys.exit(1)

if __name__ == "__main__":
    main()
