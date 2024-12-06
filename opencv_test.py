import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem
image_path = "image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calcular a proporção de pixels pretos na imagem
black_pixels = np.sum(image == 0)
total_pixels = image.size
black_ratio = black_pixels / total_pixels

print(f"Proporção de pixels pretos: {black_ratio:.2%}")

# Verificar se a maior parte da imagem contém fundo preto
if black_ratio > 0.5:  # Ajuste este valor se necessário
    print("A imagem tem fundo preto predominante. Invertendo as cores...")
    image = cv2.bitwise_not(image)

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
plt.imshow(binary_image, cmap="gray")
plt.axis("off")
plt.title("Área de Interesse Cortada")

# Salvar a imagem cortada
cv2.imwrite("cropped_image1.png", cropped_image)

