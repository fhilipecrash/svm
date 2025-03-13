import pydicom
import argparse
import cv2
import numpy as np
import os
from glob import glob

def truncation_normalization(img):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[img != 0], 5)
    Pmax = np.percentile(img[img != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)  
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[img == 0] = 0
    return normalized

def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image (0-1 range)
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img * 255, dtype=np.uint8))
    return cl

def crop_breast_region(img, photometric_interpretation):
    """
    Recorta a região da mama na imagem DICOM.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur_uint8 = (blur * 255).astype(np.uint8)
    _, breast_mask = cv2.threshold(blur_uint8, 0, 255, cv2.THRESH_BINARY if photometric_interpretation == "MONOCHROME2" else cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return img[y:y+h, x:x+w]

def segment_masses(img):
    """
    Segmenta massas na mamografia usando detecção de contornos.
    @img : numpy array image (0-255 range, uint8)
    return: máscara binária das massas
    """
    # Aplicar um filtro Gaussiano para suavizar a imagem
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Detecção de bordas usando Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Criar uma máscara vazia
    mask = np.zeros_like(img)
    
    # Desenhar os contornos na máscara
    for contour in contours:
        # Filtrar contornos pequenos (ruído)
        if cv2.contourArea(contour) > 100:  # Ajuste o valor conforme necessário
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    return mask

def mask_to_bbox(mask):
    """
    Converte uma máscara binária em uma única bounding box (da maior massa).
    @mask : numpy array (máscara binária)
    return: bounding box no formato [x_min, y_min, x_max, y_max]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Encontrar o contorno com a maior área
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return [x, y, x + w, y + h]  # [x_min, y_min, x_max, y_max]

def draw_bbox(img, bbox):
    """
    Desenha uma única bounding box na imagem.
    @img : numpy array image
    @bbox : bounding box no formato [x_min, y_min, x_max, y_max]
    return: imagem com a bounding box desenhada
    """
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Cor: verde, Espessura: 2
    return img

def load_dcm_image(file_path):
    print(f"Processando imagem DICOM: {file_path}")
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    
    # Normalização básica para 0-1
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    
    # Recorte da região da mama
    img_cropped = crop_breast_region(img, dcm_data[0x28, 0x04].value)
    
    # Normalização por percentis após o recorte
    img_normalized = truncation_normalization(img_cropped)
    
    # Aplicação do CLAHE
    cl1 = clahe(img_normalized, 1.0)
    cl2 = clahe(img_normalized, 2.0)
    
    # Combinação dos canais
    img_final = cv2.merge((
        np.array(img_normalized * 255, dtype=np.uint8),
        cl1,
        cl2
    ))
    
    # Redimensionamento final
    img_final = cv2.resize(img_final, (512, 512))
    
    # Segmentação das massas
    mass_mask = segment_masses(img_final[:, :, 0])  # Usar apenas o primeiro canal (grayscale)
    
    return img_final.astype(np.uint8), mass_mask

def main():
    parser = argparse.ArgumentParser(description="Pré-processar imagens DICOM e salvá-las em PNG.")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório contendo arquivos DICOM.")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório para salvar as imagens processadas.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Diretório para salvar as máscaras das massas.")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Diretório para salvar as imagens com bounding boxes.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.bbox_dir, exist_ok=True)
    
    for file_path in glob(os.path.join(args.input_dir, "*.dcm")):
        try:
            img, mass_mask = load_dcm_image(file_path)
            
            # Salvar a imagem processada
            output_path = os.path.join(args.output_dir, os.path.basename(file_path).replace(".dcm", ".png"))
            cv2.imwrite(output_path, img)
            print(f"Imagem salva em: {output_path}")
            
            # Salvar a máscara das massas
            mask_path = os.path.join(args.mask_dir, os.path.basename(file_path).replace(".dcm", "_mask.png"))
            cv2.imwrite(mask_path, mass_mask)
            print(f"Máscara salva em: {mask_path}")
            
            # Extrair a bounding box da maior massa
            bbox = mask_to_bbox(mass_mask)
            
            # Desenhar a bounding box na imagem
            img_with_bbox = draw_bbox(img.copy(), bbox)
            
            # Salvar a imagem com a bounding box
            bbox_path = os.path.join(args.bbox_dir, os.path.basename(file_path).replace(".dcm", "_bbox.png"))
            cv2.imwrite(bbox_path, img_with_bbox)
            print(f"Imagem com bounding box salva em: {bbox_path}")
        except Exception as e:
            print(f"Erro ao processar {file_path}: {str(e)}")

if __name__ == "__main__":
    main()
