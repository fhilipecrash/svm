import argparse
import os
import sys
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model():
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
    return model

def load_data(data_dir):
    image_files = glob(os.path.join(data_dir, "*.png"))
    if not image_files:
        print(f"Nenhuma imagem PNG encontrada no diretório: {data_dir}")
        sys.exit(1)
    labels = [0 if i % 2 == 0 else 1 for i in range(len(image_files))]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    images = np.array([img / 255.0 for img in images]).reshape(-1, 256, 256, 1)
    labels = to_categorical(labels, num_classes=2)
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

        print("Carregando dados de treinamento...")
        images, labels = load_data(train_dir)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(images)):
            print(f"Treinando Fold {fold + 1}...")
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = build_model()
            datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            datagen.fit(X_train)

            model.fit(datagen.flow(X_train, y_train, batch_size=16),
                      validation_data=(X_val, y_val),
                      epochs=10, verbose=2)

            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
            accuracies.append(val_acc)
            print(f"Fold {fold + 1} - Acurácia na validação: {val_acc:.2%}")

        print(f"Acurácia média nos folds: {np.mean(accuracies):.2%}")
        model.fit(images, labels, epochs=10, batch_size=16, verbose=2)
        val_loss, val_acc = model.evaluate(images, labels, verbose=2)
        print(f"Acurácia na validação final: {val_acc:.2%}")
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
        print(f"Acurácia nos dados de teste: {accuracy:.2%}")
        print("\nRelatório de classificação:")
        print(classification_report(y_true, y_pred))

    else:
        print("Use --train para treinamento ou --dcm para teste.")
        sys.exit(1)

if __name__ == "__main__":
    main()

