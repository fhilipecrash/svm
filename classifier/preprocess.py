import pydicom
import argparse
import cv2
import numpy as np
import os
from glob import glob
from pydicom.uid import generate_uid
from zipfile import ZipFile

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
    return img[y:y+h, x:x+w], (x, y, w, h)

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

def generate_srs(coordinates):
    """
    Gera uma SR (Structured Reporting) para uma imagem DICOM.
    """
    x, y, w, h = coordinates
    print("Gerando SR...")
    print(x, y, w, h)
    ds = pydicom.Dataset()

    # Set required DICOM attributes
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"  # SOP Class UID for SR
    ds.SOPInstanceUID = generate_uid()  # Generate a unique UID for the SR
    ds.Modality = "SR"  # Modality is Structured Report
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # Add patient and study information (example data)
    ds.PatientName = "Doe^John"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    # Step 4: Map the OpenCV bounding box to DICOM SR
    # Create a GraphicAnnotationSequence for the bounding box
    graphic_annotation = pydicom.Dataset()
    graphic_annotation.GraphicAnnotationSequence = pydicom.Sequence()

    # Define the bounding box as a rectangle in DICOM SR
    bounding_box = pydicom.Dataset()
    bounding_box.GraphicType = "RECTANGLE"  # Type of graphic (rectangle)
    bounding_box.GraphicData = [x, y, x + w, y, x + w, y + h, x, y + h, x, y]  # Rectangle coordinates
    bounding_box.GraphicAnnotationUnits = "PIXEL"  # Units for the coordinates

    # Add the bounding box to the GraphicAnnotationSequence
    graphic_annotation.GraphicAnnotationSequence.append(bounding_box)

    # Add the GraphicAnnotationSequence to the SR ContentSequence
    ds.ContentSequence = pydicom.Sequence([graphic_annotation])

    # Step 5: Save the DICOM SR file
    output_sr_path = "output_sr.dcm"
    ds.save_as(output_sr_path)

    print(f"DICOM SR saved successfully at {output_sr_path}")

def load_dcm_image(file_path):
    print(f"Processando imagem DICOM: {file_path}")
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    
    # Normalização básica para 0-1
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    
    # Recorte da região da mama
    img_cropped, coordinates = crop_breast_region(img, dcm_data.PhotometricInterpretation)
    
    # generate_srs(coordinates)

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
    
    return dcm_data, img_final, mass_mask


def main(input_file=None):
    input_dir = "zip/"

    os.makedirs("output/", exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    with ZipFile(input_file, 'r') as zip_ref:
        zip_ref.extractall(path=input_dir)
    
    output_paths = []

    for file_path in glob(os.path.join(input_dir, "*.dcm")):
        try:
            dcm_data, img_final, mass_mask = load_dcm_image(file_path)
            
            # Extrair a bounding box da maior massa
            bbox = mask_to_bbox(mass_mask)
            
            # Desenhar a bounding box na imagem
            img_with_bbox = draw_bbox(img_final.copy(), bbox)
            
            # Atualizar o PixelData do dataset DICOM com a imagem que contém a bounding box
            dcm_data.PixelData = img_with_bbox.tobytes()
            dcm_data.SOPInstanceUID = generate_uid()
            dcm_data.ImageComments= "Teste"
            dcm_data.SamplesPerPixel = 3  # Para RGB, 3 canais
            dcm_data.PhotometricInterpretation = "RGB"  # Especificando que a imagem é colorida
            dcm_data.BitsAllocated = 8  # Cada canal de cor usa 8 bits
            dcm_data.BitsStored = 8
            dcm_data.HighBit = 7  # O bit mais alto (para 8 bits)
            dcm_data.PixelRepresentation = 0  # Representação de valor positivo para cada canal
            dcm_data.WindowCenter = 128
            dcm_data.WindowWidth = 256
            dcm_data.Rows, dcm_data.Columns = img_with_bbox.shape[:2]
            
            # Salvar o dataset DICOM modificado
            output_path = os.path.join("output/", os.path.basename(file_path))
            dcm_data.save_as(output_path)
            print(f"Dataset DICOM com bounding box salvo em: {output_path}")
            output_paths.append(output_path)
        except Exception as e:
            print(f"Erro ao processar {file_path}: {str(e)}")

    print(output_paths)
    return output_paths

if __name__ == "__main__":
    main()
