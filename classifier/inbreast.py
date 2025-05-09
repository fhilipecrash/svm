import cv2
import numpy as np
import pydicom
import plistlib
import os
import glob
from matplotlib import pyplot as plt

def parse_point(point_str):
    """Converte string '(x, y)' em tupla de floats"""
    point_str = point_str.strip('()').replace(' ', '')
    x, y = map(float, point_str.split(','))
    return (x, y)

def find_matching_xml(dicom_path, xml_dir):
    """Encontra o arquivo XML correspondente ao DICOM"""
    dicom_name = os.path.basename(dicom_path)
    study_id = dicom_name.split('_')[0]  # Pega a primeira parte antes do primeiro underscore
    
    # Procurar por arquivos XML que começam com o study_id
    xml_files = glob.glob(os.path.join(xml_dir, f"{study_id}*.xml"))
    
    if xml_files:
        return xml_files[0]  # Retorna o primeiro arquivo encontrado
    return None

def process_dicom_with_rois(dicom_path, xml_path, output_dir):
    """Processa um único arquivo DICOM com seu XML correspondente"""
    try:
        # Carregar imagem DICOM
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        
        # Converter para 8 bits (se necessário) e para colorido
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Parse do XML
        with open(xml_path, 'rb') as f:
            plist_data = plistlib.load(f)
        
        # Verificar se há ROIs para esta imagem
        if 'Images' not in plist_data or len(plist_data['Images']) == 0:
            print(f"Nenhuma ROI encontrada no arquivo {xml_path}")
            return
        
        # Desenhar cada ROI
        for i, roi in enumerate(plist_data['Images'][0]['ROIs']):
            points = roi['Point_px']
            roi_name = roi['Name']
            num_points = roi['NumberOfPoints']
            
            # Converter pontos para formato numpy
            pts = np.array([parse_point(p) for p in points], dtype=np.int32)
            
            # Escolher cor baseada no tipo
            if roi_name == 'Mass':
                color = (0, 0, 255)  # Vermelho para massas
                thickness = 3
                alpha = 0.3  # Transparência para preenchimento
            else:
                color = (255, 0, 0)  # Azul para calcificações
                thickness = 1 if num_points > 1 else 3
                alpha = 0.7 if num_points > 1 else 1.0
            
            # Criar uma cópia para desenho com transparência
            overlay = img_color.copy()
            
            # Desenhar
            if num_points == 1:
                # Ponto único - desenhar um círculo
                center = tuple(pts[0])
                cv2.circle(overlay, center, 5, color, thickness)
            else:
                # Polígono - desenhar contorno e preenchimento
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=thickness)
                
                # Adicionar número da ROI
                centroid = np.mean(pts, axis=0).astype(int)[0]
                cv2.putText(overlay, str(i), tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Aplicar transparência
            img_color = cv2.addWeighted(overlay, alpha, img_color, 1 - alpha, 0)
        
        # Salvar a imagem resultante
        output_filename = os.path.basename(dicom_path).replace('.dcm', '_annotated.png')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img_color)
        print(f"Imagem com ROIs salva em {output_path}")
        
        # Mostrar a imagem (opcional - comente se estiver processando muitos arquivos)
        # plt.figure(figsize=(15, 15))
        # plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        # plt.title(f'ROIs na Mamografia - {os.path.basename(dicom_path)}')
        # plt.axis('off')
        # plt.show()
        
        return True
    
    except Exception as e:
        print(f"Erro ao processar {dicom_path}: {str(e)}")
        return False

def process_all_dicoms(input_dir, xml_dir, output_dir):
    """Processa todos os arquivos DICOM no diretório de entrada"""
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Listar todos os arquivos DICOM no diretório de entrada
    dicom_files = glob.glob(os.path.join(input_dir, "*.dcm"))
    
    if not dicom_files:
        print(f"Nenhum arquivo DICOM encontrado em {input_dir}")
        return
    
    print(f"Encontrados {len(dicom_files)} arquivos DICOM para processamento")
    
    processed_count = 0
    skipped_count = 0
    
    for dicom_path in dicom_files:
        # Encontrar arquivo XML correspondente
        xml_path = find_matching_xml(dicom_path, xml_dir)
        
        if xml_path is None:
            print(f"Nenhum arquivo XML encontrado para {os.path.basename(dicom_path)} - pulando")
            skipped_count += 1
            continue
        
        print(f"\nProcessando {os.path.basename(dicom_path)} com {os.path.basename(xml_path)}")
        
        # Processar o arquivo DICOM
        success = process_dicom_with_rois(dicom_path, xml_path, output_dir)
        
        if success:
            processed_count += 1
        else:
            skipped_count += 1
    
    print(f"\nProcessamento concluído:")
    print(f"- Arquivos processados com sucesso: {processed_count}")
    print(f"- Arquivos sem XML correspondente: {skipped_count}")

# Exemplo de uso
if __name__ == "__main__":
    # Configurações (ajuste conforme necessário)
    input_dir = "./inbreast/AllDICOMs"  # Diretório contendo os arquivos DICOM
    xml_dir = "./inbreast/AllXML"    # Diretório contendo os arquivos XML
    output_dir = "./annotated_images"
    
    process_all_dicoms(input_dir, xml_dir, output_dir)