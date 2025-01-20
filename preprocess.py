import pydicom
import argparse
import cv2
import cupy as cp
import numpy as np
import os
from glob import glob

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
    print(f"Processando imagem DICOM: {file_path}")
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    img = anisotropic_diffusion_with_median_filter_gpu(img)
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
