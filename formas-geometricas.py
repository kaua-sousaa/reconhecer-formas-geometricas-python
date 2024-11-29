import cv2
import numpy as np

def identificar_forma_e_cor(imagem):
    img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(img_cinza, 50, 150)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #lista das coordenadas dos vertices
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        
        if area < 100:
            continue
        
        perimetro = cv2.arcLength(contorno, True)
        tolerancia = 0.04 * perimetro
        vertices = cv2.approxPolyDP(contorno, tolerancia, True)
        num_vertices = len(vertices)

        if num_vertices == 3:
            forma = "Triangulo"
        elif num_vertices == 4:
            x, y, largura, altura = cv2.boundingRect(vertices)
            proporcao = largura / float(altura)
            if proporcao >= 0.9 and proporcao <= 1.1:
                forma = "Quadrado"
            else:
                forma = "Retangulo"
        else:
            circularidade = (4 * np.pi * area) / (perimetro ** 2)
            if circularidade > 0.85:
                forma = "Circulo"
            else:
                forma = "Poligono Irregular"
        
        mascara = np.zeros(imagem.shape[:2], dtype="uint8")
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        media_cor = cv2.mean(imagem, mask=mascara)
        cor = determinar_cor(media_cor)

        cv2.drawContours(imagem, [vertices], -1, (0, 255, 0), 2)
        x, y, _, _ = cv2.boundingRect(vertices)
        texto = f"{forma} - {cor}"
        cv2.putText(imagem, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return imagem

def determinar_cor(media_cor):
    azul, verde, vermelho = media_cor[:3]
    if vermelho > 200 and verde < 100 and azul < 100:
        return "Vermelho"
    elif verde > 200 and vermelho < 100 and azul < 100:
        return "Verde"
    elif azul > 200 and vermelho < 100 and verde < 100:
        return "Azul"
    elif vermelho > 200 and verde > 200 and azul < 100:
        return "Amarelo"
    elif vermelho > 200 and verde > 200 and azul > 200:
        return "Branco"
    elif vermelho < 50 and verde < 50 and azul < 50:
        return "Preto"
    else:
        return "Cor desconhecida"

# Carregar e processar a imagem
caminho_imagem = 'FIGURA.png'
imagem = cv2.imread(caminho_imagem)
imagem_com_formas = identificar_forma_e_cor(imagem)

# Exibir resultados
cv2.imshow("Formas e Cores Identificadas", imagem_com_formas)
cv2.waitKey(0)
cv2.destroyAllWindows()
