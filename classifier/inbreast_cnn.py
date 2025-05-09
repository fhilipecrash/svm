import cv2
import numpy as np
import pydicom
import plistlib
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Funções de Pré-processamento
def parse_point(point_str):
    """Converte string '(x, y)' em tupla de floats"""
    point_str = point_str.strip('()').replace(' ', '')
    x, y = map(float, point_str.split(','))
    return (x, y)

def find_matching_xml(dicom_path, xml_dir):
    """Encontra o arquivo XML correspondente ao DICOM"""
    dicom_name = os.path.basename(dicom_path)
    study_id = dicom_name.split('_')[0]
    xml_files = glob.glob(os.path.join(xml_dir, f"{study_id}*.xml"))
    return xml_files[0] if xml_files else None

def create_mask_from_rois(dicom_path, xml_path, output_size):
    """Cria máscara binária a partir das anotações XML"""
    try:
        # Carregar DICOM para obter dimensões originais
        ds = pydicom.dcmread(dicom_path)
        original_shape = ds.pixel_array.shape
        
        # Criar máscara vazia no tamanho original
        mask = np.zeros(original_shape, dtype=np.uint8)
        
        # Parse do XML
        with open(xml_path, 'rb') as f:
            plist_data = plistlib.load(f)
        
        # Desenhar ROIs na máscara
        if 'Images' in plist_data and len(plist_data['Images']) > 0:
            for roi in plist_data['Images'][0]['ROIs']:
                points = roi['Point_px']
                pts = np.array([parse_point(p) for p in points], dtype=np.int32)
                
                if len(pts) == 1:  # Ponto único
                    cv2.circle(mask, tuple(pts[0]), 5, 255, -1)
                else:  # Polígono
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
        
        # Redimensionar para o tamanho desejado
        mask = cv2.resize(mask, output_size)
        return mask
    
    except Exception as e:
        print(f"Erro ao processar {dicom_path}: {str(e)}")
        return None

def load_dicom_image(dicom_path, output_size):
    """Carrega e pré-processa imagem DICOM"""
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    
    # Normalizar para 8-bit
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Redimensionar
    img = cv2.resize(img, output_size)
    return img

def prepare_dataset(dicom_dir, xml_dir, output_size=(512, 512)):
    """Prepara dataset de imagens e máscaras"""
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    X, y = [], []
    
    for dicom_path in dicom_files:
        xml_path = find_matching_xml(dicom_path, xml_dir)
        if xml_path:
            img = load_dicom_image(dicom_path, output_size)
            mask = create_mask_from_rois(dicom_path, xml_path, output_size)
            
            if mask is not None:
                X.append(img)
                y.append(mask)
    
    # Converter para arrays numpy e adicionar dimensão de canal
    X = np.expand_dims(np.array(X), axis=-1).astype('float32') / 255.0
    y = np.expand_dims(np.array(y), axis=-1).astype('float32') / 255.0
    
    return X, y

# 2. Arquitetura da U-Net
def build_unet(input_shape=(512, 512, 1)):
    """Constrói modelo U-Net para segmentação"""
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Centro
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Saída
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 3. Treinamento do Modelo

def train_model(X_train, y_train, X_val, y_val, model_path='unet_mammo.h5'):
    """Treina o modelo U-Net com gerador de dados correto"""
    # Configuração do aumento de dados
    data_gen_args = dict(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant'
    )
    
    # Criar ImageDataGenerator para imagens e máscaras
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Seed para garantir correspondência entre imagens e máscaras
    seed = 42
    
    # Configurar geradores
    image_generator = image_datagen.flow(
        X_train, 
        batch_size=8,
        seed=seed
    )
    
    mask_generator = mask_datagen.flow(
        y_train,
        batch_size=8,
        seed=seed
    )
    
    # Combinar geradores corretamente
    train_generator = (pair for pair in zip(image_generator, mask_generator))
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5, verbose=1),
        ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    ]
    
    # Compilar modelo
    model = build_unet(input_shape=X_train.shape[1:])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 8,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return model, history

# 4. Predição e Visualização
def predict_and_mark_rois(model, dicom_path, output_size=(512, 512), threshold=0.5):
    """Faz predição e marca ROIs na imagem original"""
    # Carregar e pré-processar imagem
    original_img = load_dicom_image(dicom_path, output_size)
    original_shape = original_img.shape
    
    # Preparar entrada para o modelo
    img_input = np.expand_dims(original_img, axis=(0, -1)).astype('float32') / 255.0
    
    # Fazer predição
    pred = model.predict(img_input)[0]
    pred_mask = (pred > threshold).astype(np.uint8) * 255
    
    # Pós-processamento
    pred_mask = postprocess_mask(pred_mask)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar na imagem original
    marked_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Simplificar contorno
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Desenhar polígono
        cv2.drawContours(marked_img, [approx], -1, (0, 255, 0), 2)
        
        # Calcular centroide para texto
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(marked_img, "ROI", (cX-20, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return original_img, pred_mask, marked_img

def postprocess_mask(mask, kernel_size=3):
    """Aplica operações morfológicas para limpar a máscara"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remover pequenos ruídos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Preencher pequenos buracos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def visualize_results(original, prediction, marked):
    """Visualiza os resultados lado a lado"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title('Predição')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    plt.title('Marcações')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. Fluxo Principal
def main():
    # Configurações
    DICOM_DIR = "./inbreast/AllDICOMs"
    XML_DIR = "./inbreast/AllXML"
    MODEL_PATH = "unet_mammo.h5"
    OUTPUT_SIZE = (512, 512)  # Tamanho para redimensionamento
    
    # 1. Preparar dataset
    print("Preparando dataset...")
    X, y = prepare_dataset(DICOM_DIR, XML_DIR, OUTPUT_SIZE)
    
    if len(X) == 0:
        print("Nenhum dado encontrado. Verifique os diretórios.")
        return
    
    # 2. Dividir em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset preparado: {len(X_train)} treino, {len(X_val)} validação")
    
    # 3. Treinar modelo (ou carregar se já existir)
    if not os.path.exists(MODEL_PATH):
        print("Treinando modelo...")
        model, history = train_model(X_train, y_train, X_val, y_val, MODEL_PATH)
    else:
        print("Carregando modelo pré-treinado...")
        model = build_unet(input_shape=X_train.shape[1:])
        model.load_weights(MODEL_PATH)
    
    # 4. Testar em uma nova imagem
    test_dicom = os.path.join(DICOM_DIR, os.listdir(DICOM_DIR)[0])
    print(f"\nTestando em: {test_dicom}")
    
    original, pred_mask, marked_img = predict_and_mark_rois(model, test_dicom, OUTPUT_SIZE)
    visualize_results(original, pred_mask, marked_img)
    
    # 5. Salvar imagem com marcações
    output_img_path = test_dicom.replace('.dcm', '_marked.jpg')
    cv2.imwrite(output_img_path, marked_img)
    print(f"Imagem com marcações salva em: {output_img_path}")

if __name__ == "__main__":
    main()