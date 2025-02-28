import cv2
import numpy as np

# Carregar a imagem
# image_path = "img/B4_D38187FE.png"
image_path = "img/B5_7C97E76F.png"
image = cv2.imread(image_path)

# Converter para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar um desfoque para reduzir ruídos
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detectar círculos usando a Transformada de Hough
circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1.6, 
    minDist=50, 
    param1=50, 
    param2=30, 
    minRadius=30, 
    maxRadius=150
)

# Verifica se algum círculo foi detectado
if circles is not None:
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]  # Pegamos o primeiro círculo encontrado
    
    # Criar um recorte da região circular detectada
    roi = image[y-r:y+r, x-r:x+r]

    # Salvar a imagem recortada
    output_path = "roi_cropped.png"
    cv2.imwrite(output_path, roi)
    print(f"Imagem da região de interesse salva em: {output_path}")
else:
    print("Nenhum círculo detectado.")
