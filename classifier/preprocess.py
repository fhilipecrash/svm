import pydicom
import argparse
import cv2
import numpy as np
import os
from glob import glob
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, EnhancedSRStorage
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
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

def generate_srs(coordinates, dcm_data):
    """
    Gera uma SR (Structured Reporting) para uma imagem DICOM com marcações de ROI.
    
    Args:
        coordinates: Tupla (x, y, w, h) com as coordenadas da ROI
        dcm_data: Dataset DICOM da imagem original
        
    Returns:
        Dataset DICOM SR com as marcações de ROI
    """
    x, y, w, h = coordinates
    print("Gerando SR para ROI...")
    
    # Criar dataset SR
    ds = Dataset()
    
    # File Meta Information
    ds.file_meta = Dataset()
    ds.file_meta.FileMetaInformationGroupLength = 192
    ds.file_meta.FileMetaInformationVersion = b'\x00\x01'
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR Storage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = '1.2.40.0.13.1.3'  # Exemplo de Implementation Class UID
    ds.file_meta.ImplementationVersionName = 'PYDICOM_SR_TOOL'
    
    # Main dataset attributes
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR Storage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.AccessionNumber = ''
    ds.Modality = 'SR'
    ds.Manufacturer = 'PYDICOM SR TOOL'
    ds.StudyDate = dcm_data.get('StudyDate', '')
    ds.SeriesDate = dcm_data.get('SeriesDate', '')
    ds.ContentDate = dcm_data.get('ContentDate', '')
    ds.StudyTime = dcm_data.get('StudyTime', '')
    ds.SeriesTime = dcm_data.get('SeriesTime', '')
    ds.ContentTime = dcm_data.get('ContentTime', '')
    ds.ReferringPhysicianName = ''
    ds.SeriesDescription = 'Research Derived series'
    ds.ManufacturerModelName = 'PYDICOM SR'
    
    # Patient information
    ds.PatientName = getattr(dcm_data, 'PatientName', '')
    ds.PatientID = getattr(dcm_data, 'PatientID', '')
    ds.IssuerOfPatientID = getattr(dcm_data, 'IssuerOfPatientID', '')
    ds.PatientBirthDate = getattr(dcm_data, 'PatientBirthDate', '')
    ds.PatientSex = getattr(dcm_data, 'PatientSex', '')
    ds.PatientAge = getattr(dcm_data, 'PatientAge', '')
    
    # Study information
    ds.StudyInstanceUID = getattr(dcm_data, 'StudyInstanceUID', '')
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = getattr(dcm_data, 'StudyID', '')
    ds.SeriesNumber = '99'
    ds.InstanceNumber = '1'
    
    # Content information
    ds.ContentQualification = 'RESEARCH'
    ds.ValueType = 'CONTAINER'
    ds.ContinuityOfContent = 'SEPARATE'
    ds.CompletionFlag = 'COMPLETE'
    ds.VerificationFlag = 'UNVERIFIED'
    
    # Content Template Sequence
    content_template = Dataset()
    content_template.MappingResource = 'DCMR'
    content_template.TemplateIdentifier = '1500'
    ds.ContentTemplateSequence = pydicom.Sequence([content_template])
    
    # Concept Name Code Sequence (Report title)
    concept_name = Dataset()
    concept_name.CodeValue = '126000'
    concept_name.CodingSchemeDesignator = 'DCM'
    concept_name.CodeMeaning = 'Imaging Measurement Report'
    ds.ConceptNameCodeSequence = pydicom.Sequence([concept_name])
    
    # Current Requested Procedure Evidence Sequence
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = getattr(dcm_data, 'SeriesInstanceUID', '')
    
    ref_sop = Dataset()
    ref_sop.ReferencedSOPClassUID = getattr(dcm_data, 'SOPClassUID', '')
    ref_sop.ReferencedSOPInstanceUID = dcm_data.SOPInstanceUID
    
    ref_sop_seq = Dataset()
    ref_sop_seq.ReferencedSOPSequence = pydicom.Sequence([ref_sop])
    
    ref_series.ReferencedSOPSequence = pydicom.Sequence([ref_sop_seq])
    
    req_proc_evidence = Dataset()
    req_proc_evidence.ReferencedSeriesSequence = pydicom.Sequence([ref_series])
    req_proc_evidence.StudyInstanceUID = ds.StudyInstanceUID
    
    ds.CurrentRequestedProcedureEvidenceSequence = pydicom.Sequence([req_proc_evidence])
    
    # Content Sequence - Main structure
    content_seq = []
    
    # Language of Content
    language_item = Dataset()
    language_item.RelationshipType = 'HAS CONCEPT MOD'
    language_item.ValueType = 'CODE'
    
    language_name = Dataset()
    language_name.CodeValue = '121049'
    language_name.CodingSchemeDesignator = 'DCM'
    language_name.CodeMeaning = 'Language of Content Item and Descendants'
    language_item.ConceptNameCodeSequence = pydicom.Sequence([language_name])
    
    language_code = Dataset()
    language_code.CodeValue = 'eng'
    language_code.CodingSchemeDesignator = 'RFC5646'
    language_code.CodeMeaning = 'English'
    language_item.ConceptCodeSequence = pydicom.Sequence([language_code])
    
    # Country of Language
    country_item = Dataset()
    country_item.RelationshipType = 'HAS CONCEPT MOD'
    country_item.ValueType = 'CODE'
    
    country_name = Dataset()
    country_name.CodeValue = '121046'
    country_name.CodingSchemeDesignator = 'DCM'
    country_name.CodeMeaning = 'Country of Language'
    country_item.ConceptNameCodeSequence = pydicom.Sequence([country_name])
    
    country_code = Dataset()
    country_code.CodeValue = 'US'
    country_code.CodingSchemeDesignator = 'ISO3166_1'
    country_code.CodeMeaning = 'United States'
    country_item.ConceptCodeSequence = pydicom.Sequence([country_code])
    
    language_item.ContentSequence = pydicom.Sequence([country_item])
    content_seq.append(language_item)
    
    # Person Observer Name
    observer_item = Dataset()
    observer_item.RelationshipType = 'HAS OBS CONTEXT'
    observer_item.ValueType = 'PNAME'
    
    observer_name = Dataset()
    observer_name.CodeValue = '121008'
    observer_name.CodingSchemeDesignator = 'DCM'
    observer_name.CodeMeaning = 'Person Observer Name'
    observer_item.ConceptNameCodeSequence = pydicom.Sequence([observer_name])
    
    observer_item.PersonName = 'unknown^unknown'
    content_seq.append(observer_item)
    
    # Procedure Reported
    procedure_item = Dataset()
    procedure_item.RelationshipType = 'HAS CONCEPT MOD'
    procedure_item.ValueType = 'CODE'
    
    procedure_name = Dataset()
    procedure_name.CodeValue = '121058'
    procedure_name.CodingSchemeDesignator = 'DCM'
    procedure_name.CodeMeaning = 'Procedure reported'
    procedure_item.ConceptNameCodeSequence = pydicom.Sequence([procedure_name])
    
    procedure_code = Dataset()
    procedure_code.CodeValue = '1'
    procedure_code.CodingSchemeDesignator = '99dcmjs'
    procedure_code.CodeMeaning = 'Unknown procedure'
    procedure_item.ConceptCodeSequence = pydicom.Sequence([procedure_code])
    
    content_seq.append(procedure_item)
    
    # Image Library Container
    image_library = Dataset()
    image_library.RelationshipType = 'CONTAINS'
    image_library.ValueType = 'CONTAINER'
    
    image_library_name = Dataset()
    image_library_name.CodeValue = '111028'
    image_library_name.CodingSchemeDesignator = 'DCM'
    image_library_name.CodeMeaning = 'Image Library'
    image_library.ConceptNameCodeSequence = pydicom.Sequence([image_library_name])
    
    image_library.ContinuityOfContent = 'SEPARATE'
    
    # Image Library Group
    image_group = Dataset()
    image_group.RelationshipType = 'CONTAINS'
    image_group.ValueType = 'CONTAINER'
    
    image_group_name = Dataset()
    image_group_name.CodeValue = '126200'
    image_group_name.CodingSchemeDesignator = 'DCM'
    image_group_name.CodeMeaning = 'Image Library Group'
    image_group.ConceptNameCodeSequence = pydicom.Sequence([image_group_name])
    
    image_group.ContinuityOfContent = 'SEPARATE'
    
    # Referenced Image
    ref_image = Dataset()
    ref_image.RelationshipType = 'CONTAINS'
    ref_image.ValueType = 'IMAGE'
    
    ref_sop_item = Dataset()
    ref_sop_item.ReferencedSOPClassUID = getattr(dcm_data, 'SOPClassUID', '')
    ref_sop_item.ReferencedSOPInstanceUID = getattr(dcm_data, 'SOPInstanceUID', '')
    ref_image.ReferencedSOPSequence = pydicom.Sequence([ref_sop_item])
    
    image_group.ContentSequence = pydicom.Sequence([ref_image])
    image_library.ContentSequence = pydicom.Sequence([image_group])
    content_seq.append(image_library)
    
    # Imaging Measurements Container
    measurements = Dataset()
    measurements.RelationshipType = 'CONTAINS'
    measurements.ValueType = 'CONTAINER'
    
    measurements_name = Dataset()
    measurements_name.CodeValue = '126010'
    measurements_name.CodingSchemeDesignator = 'DCM'
    measurements_name.CodeMeaning = 'Imaging Measurements'
    measurements.ConceptNameCodeSequence = pydicom.Sequence([measurements_name])
    
    measurements.ContinuityOfContent = 'SEPARATE'
    
    # Measurement Group
    measurement_group = Dataset()
    measurement_group.RelationshipType = 'CONTAINS'
    measurement_group.ValueType = 'CONTAINER'
    
    measurement_group_name = Dataset()
    measurement_group_name.CodeValue = '125007'
    measurement_group_name.CodingSchemeDesignator = 'DCM'
    measurement_group_name.CodeMeaning = 'Measurement Group'
    measurement_group.ConceptNameCodeSequence = pydicom.Sequence([measurement_group_name])
    
    measurement_group.ContinuityOfContent = 'SEPARATE'
    
    # Tracking Identifier
    tracking_id = Dataset()
    tracking_id.RelationshipType = 'HAS OBS CONTEXT'
    tracking_id.ValueType = 'TEXT'
    
    tracking_id_name = Dataset()
    tracking_id_name.CodeValue = '112039'
    tracking_id_name.CodingSchemeDesignator = 'DCM'
    tracking_id_name.CodeMeaning = 'Tracking Identifier'
    tracking_id.ConceptNameCodeSequence = pydicom.Sequence([tracking_id_name])
    
    tracking_id.TextValue = 'cornerstoneTools@^4.0.0:Rectangle'
    measurement_group.ContentSequence = pydicom.Sequence([tracking_id])
    
    # Tracking UID
    tracking_uid = Dataset()
    tracking_uid.RelationshipType = 'HAS OBS CONTEXT'
    tracking_uid.ValueType = 'UIDREF'
    
    tracking_uid_name = Dataset()
    tracking_uid_name.CodeValue = '112040'
    tracking_uid_name.CodingSchemeDesignator = 'DCM'
    tracking_uid_name.CodeMeaning = 'Tracking Unique Identifier'
    tracking_uid.ConceptNameCodeSequence = pydicom.Sequence([tracking_uid_name])
    
    tracking_uid.UID = generate_uid()
    measurement_group.ContentSequence.append(tracking_uid)
    
    # ROI Graphic Data
    graphic_item = Dataset()
    graphic_item.RelationshipType = 'CONTAINS'
    graphic_item.ValueType = 'SCOORD'
    
    # Graphic Data for Rectangle (x1, y1, x2, y2, x3, y3, x4, y4, x1, y1)
    graphic_item.GraphicData = [
        x, y,          # Top-left
        x + w, y,      # Top-right
        x + w, y + h,  # Bottom-right
        x, y + h,      # Bottom-left
        x, y           # Back to top-left to close
    ]
    graphic_item.GraphicType = 'POLYGON'
    
    # Referenced Image for Graphic
    ref_graphic = Dataset()
    ref_graphic.RelationshipType = 'SELECTED FROM'
    ref_graphic.ValueType = 'IMAGE'
    
    ref_sop_graphic = Dataset()
    ref_sop_graphic.ReferencedSOPClassUID = getattr(dcm_data, 'SOPClassUID', '')
    ref_sop_graphic.ReferencedSOPInstanceUID = dcm_data.SOPInstanceUID
    ref_graphic.ReferencedSOPSequence = pydicom.Sequence([ref_sop_graphic])
    
    graphic_item.ContentSequence = pydicom.Sequence([ref_graphic])
    measurement_group.ContentSequence.append(graphic_item)
    
    measurements.ContentSequence = pydicom.Sequence([measurement_group])
    content_seq.append(measurements)
    
    ds.ContentSequence = pydicom.Sequence(content_seq)
    
    return ds

def load_dcm_image(file_path):
    print(f"Processando imagem DICOM: {file_path}")
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array.astype(float)
    
    # Normalização básica para 0-1
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    
    # Recorte da região da mama
    img_cropped, coordinates = crop_breast_region(img, dcm_data.PhotometricInterpretation)

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
    # img_final = cv2.resize(img_final, (512, 512))
    
    # Segmentação das massas
    mass_mask = segment_masses(img_final[:, :, 0])  # Usar apenas o primeiro canal (grayscale)
    
    return dcm_data, img_final, mass_mask, coordinates


def main(input_file=None):
    input_dir = "zip/"

    os.makedirs("output/", exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    with ZipFile(input_file, 'r') as zip_ref:
        zip_ref.extractall(path=input_dir)
    
    output_paths = []

    for file_path in glob(os.path.join(input_dir, "*.dcm")):
        try:
            dcm_data, img_final, mass_mask, coordinates = load_dcm_image(file_path)
            
            # Extrair a bounding box da maior massa
            bbox = mask_to_bbox(mass_mask)
            
            # Desenhar a bounding box na imagem
            img_with_bbox = draw_bbox(img_final.copy(), bbox)
            
            # Atualizar o PixelData do dataset DICOM com a imagem que contém a bounding box
            dcm_data.PixelData = img_with_bbox.tobytes()
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
            # dcm_data.save_as(output_path)
            ds = generate_srs(coordinates, dcm_data)
            ds.save_as(output_path)
            print(f"Dataset DICOM com bounding box salvo em: {output_path}")
            output_paths.append(output_path)
        except Exception as e:
            print(f"Erro ao processar {file_path}: {str(e)}")

    return output_paths

if __name__ == "__main__":
    main()
