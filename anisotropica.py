import numpy as np
import cupy as cp
import pydicom
import cv2
from matplotlib import pyplot as plt

def anisotropic_diffusion_with_median_filter_gpu(img, num_iter=5, kappa=50, gamma=0.1):
    """
    Aplica difusão anisotrópica com filtro mediano em uma imagem.
    """
    # Transfere a imagem para a GPU
    img_gpu = cp.array(img, dtype=cp.float32)

    # Realiza a difusão anisotrópica
    for i in range(num_iter):
        dx = cp.gradient(img_gpu, axis=1)
        dy = cp.gradient(img_gpu, axis=0)
        grad_magnitude = cp.sqrt(dx ** 2 + dy ** 2)
        c = cp.exp(-(grad_magnitude / kappa) ** 2)
        img_gpu += gamma * (c * dx + c * dy)

    # Converte a imagem de volta para CPU
    img_cpu = cp.asnumpy(img_gpu)

    # Aplica o filtro mediano
    img_cpu = cv2.medianBlur(img_cpu.astype(np.float32), 3)

    return img_cpu

def load_dcm_image(file_path):
    """
    Carrega e normaliza a imagem DICOM.
    """
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    return img

def show_images(original, processed):
    """
    Mostra a imagem original e a tratada lado a lado.
    """
    plt.figure(figsize=(12, 6))

    # Imagem original
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis("off")

    # Imagem processada
    plt.subplot(1, 2, 2)
    plt.title("Imagem Processada")
    plt.imshow(processed, cmap='gray')
    plt.axis("off")

    plt.savefig("anisotropica.png")

if __name__ == "__main__":
    dicom_file = "B5_FED9C57C.dcm"  # Substitua pelo caminho do arquivo DICOM
    original_image = load_dcm_image(dicom_file)
    processed_image = anisotropic_diffusion_with_median_filter_gpu(original_image)

    # Mostra o antes e o depois
    show_images(original_image, processed_image)
