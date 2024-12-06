import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem
image_path = "image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar o limiar adaptativo para segmentar a região de interesse
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Selecionar o maior contorno
largest_contour = max(contours, key=cv2.contourArea)

# Obter o retângulo delimitador da área de interesse
x, y, w, h = cv2.boundingRect(largest_contour)

# Fazer o corte na forma de um quadrado
side_length = max(w, h)
center_x = x + w // 2
center_y = y + h // 2

start_x = max(center_x - side_length // 2, 0)
start_y = max(center_y - side_length // 2, 0)

end_x = min(center_x + side_length // 2, image.shape[1])
end_y = min(center_y + side_length // 2, image.shape[0])

# Cortar a área de interesse
cropped_image = image[start_y:end_y, start_x:end_x]

# Exibir o resultado
plt.figure(figsize=(8, 8))
plt.imshow(cropped_image, cmap="gray")
plt.axis("off")
plt.title("Área de Interesse Cortada")

# Salvar a imagem cortada
output_path = "cropped_image3.png"
cv2.imwrite(output_path, cropped_image)
print(f"A imagem cortada foi salva em: {output_path}")
