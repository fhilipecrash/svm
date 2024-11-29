import matplotlib.pyplot as plt
from skimage import data, color
import numpy as np
import scipy.ndimage

def anisotropic_diffusion(image, num_iterations, kappa, gamma=0.2, option=1):
    """
    Aplica difusão anisotrópica em uma imagem.

    Parâmetros:
        image (ndarray): Imagem de entrada.
        num_iterations (int): Número de iterações.
        kappa (float): Parâmetro de condução (controla a sensibilidade ao gradiente).
        gamma (float): Taxa de atualização (normalmente entre 0 e 0.25 para estabilidade).
        option (int): Escolha da função de condução:
                      1: exp(-(gradient/kappa)^2)
                      2: 1 / (1 + (gradient/kappa)^2)
    Retorna:
        ndarray: Imagem processada após difusão anisotrópica.
    """
    img = image.astype(np.float32)

    # Máscaras de derivadas
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])
    
    for t in range(num_iterations):
        # Calcula os gradientes
        nablaN = scipy.ndimage.convolve(img, dy, mode='nearest')
        nablaS = -scipy.ndimage.convolve(img, -dy, mode='nearest')
        nablaW = scipy.ndimage.convolve(img, dx, mode='nearest')
        nablaE = -scipy.ndimage.convolve(img, -dx, mode='nearest')
        
        # Calcula a função de condução
        if option == 1:
            cN = np.exp(-(nablaN / kappa) ** 2)
            cS = np.exp(-(nablaS / kappa) ** 2)
            cW = np.exp(-(nablaW / kappa) ** 2)
            cE = np.exp(-(nablaE / kappa) ** 2)
        elif option == 2:
            cN = 1 / (1 + (nablaN / kappa) ** 2)
            cS = 1 / (1 + (nablaS / kappa) ** 2)
            cW = 1 / (1 + (nablaW / kappa) ** 2)
            cE = 1 / (1 + (nablaE / kappa) ** 2)
        
        # Atualiza a imagem
        img += gamma * (
            cN * nablaN + cS * nablaS +
            cW * nablaW + cE * nablaE
        )
    
    return img

# Carregar uma imagem de exemplo
image = color.rgb2gray(data.astronaut())  # Converter para tons de cinza

# Aplicar difusão anisotrópica
processed_image = anisotropic_diffusion(image, num_iterations=15, kappa=30, gamma=0.2, option=1)

# Visualizar a imagem original e a processada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Processada")
plt.imshow(processed_image, cmap='gray')
plt.axis('off')

plt.savefig('anisotropic_diffusion.png')
