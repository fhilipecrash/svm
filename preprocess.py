import pydicom
import argparse
import cv2
import cupy as cp
import numpy as np
import os
from glob import glob

def normalize_image(img):
    """
    Normaliza a imagem truncando os valores entre os percentis 5 e 99 e escalando para o intervalo [0,1].
    """
    img_gpu = cp.array(img, dtype=cp.float32)
    
    # Calcula os percentis 5 e 99 na GPU
    Pmin = cp.percentile(img_gpu[img_gpu != 0], 5)
    Pmax = cp.percentile(img_gpu[img_gpu != 0], 99)
    
    # Trunca os valores para o intervalo [Pmin, Pmax]
    img_gpu = cp.clip(img_gpu, Pmin, Pmax)
    
    # Normaliza para o intervalo [0, 1]
    img_gpu = (img_gpu - Pmin) / (Pmax - Pmin)
    
    # Converte de volta para CPU e aplica filtro de desfoque
    img_cpu = cp.asnumpy(img_gpu)
    img_cpu[img == 0] = 0  # Mantém os pixels de fundo como 0
    img_cpu = cv2.blur(img_cpu, (10, 10))
    
    return img_cpu

def crop_breast_region(img, photometric_interpretation):
    """
    Recorta a região da mama na imagem DICOM.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur_uint8 = (blur * 255).astype(np.uint8)  # Converte para uint8
    _, breast_mask = cv2.threshold(blur_uint8, 0, 255, cv2.THRESH_BINARY if photometric_interpretation == "MONOCHROME2" else cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # Retorna a imagem original se não encontrar contornos
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return img[y:y+h, x:x+w]

def load_dcm_image(file_path):
    print(f"Processando imagem DICOM: {file_path}")
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    img = normalize_image(img)
    img = crop_breast_region(img, dcm_data[0x28, 0x04].value)
    img = cv2.resize(img, (256, 256))
    return (img * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Pré-processar imagens DICOM e salvá-las em PNG.")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório contendo arquivos DICOM.")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório para salvar as imagens processadas.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dicom_files = glob(os.path.join(input_dir, "*.dcm"))
    if not dicom_files:
        print("Nenhum arquivo DICOM encontrado.")
        return

    for file_path in dicom_files:
        img = load_dcm_image(file_path)
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace(".dcm", ".png"))
        cv2.imwrite(output_path, img)
        print(f"Imagem salva em: {output_path}")

if __name__ == "__main__":
    main()
