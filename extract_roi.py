import cv2
import numpy as np

def process_mammogram(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar bordas usando Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar variável para o contorno do garfo
    fork_contour = None

    # Iterar pelos contornos encontrados
    for contour in contours:
        # Aproximar o contorno
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Verificar condições (exemplo: tamanho e número de vértices)
        if len(approx) >= 4:  # O formato do garfo tem múltiplos vértices
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.5 < aspect_ratio < 3.0:  # Ajuste o intervalo conforme necessário
                fork_contour = contour
                break

    if fork_contour is None:
        print("Garfo não encontrado.")
        return

    # Criar máscara para a área interna do garfo
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [fork_contour], -1, 255, thickness=cv2.FILLED)

    # Extrair a área de interesse (dentro do garfo)
    region_of_interest = cv2.bitwise_and(image, image, mask=mask)

    # Salvar e exibir resultados
    cv2.imshow("Original", image)
    cv2.imshow("Detecção de bordas", edges)
    cv2.imshow("Área de interesse", region_of_interest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chamar a função com o caminho da imagem processada
process_mammogram('marked/B5_6E410A3E.dcm.png')
