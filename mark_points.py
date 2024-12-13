import cv2
import numpy as np
import os
from glob import glob

# Carregar a imagem
def mark_points(image_path):
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. Realce de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    # 2. Aplicar detecção de bordas para encontrar regiões suspeitas
    edges = cv2.Canny(enhanced_image, threshold1=50, threshold2=150)

    # 3. Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma cópia da imagem original para desenhar os pontos marcados
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 4. Filtrar e marcar regiões suspeitas
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1000:  # Filtrar contornos com base na área
            # Obter o círculo delimitador da região
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Desenhar um círculo ao redor da região suspeita
            cv2.circle(output_image, center, 2, (0, 0, 255), -1)

    # Salvar e exibir o resultado
    output_path = f'marked/{os.path.basename(image_path)}'
    cv2.imwrite(output_path, output_image)

    # cv2.imshow("Regiões Suspeitas", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"Imagem com pontos marcados salva em: {output_path}")

dicom_files = glob(os.path.join('img/', "*.png"))

for file in dicom_files:
    mark_points(file)