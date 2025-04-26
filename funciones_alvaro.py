#########################################################################################
# Paquetes
#########################################################################################

import cv2
import os
import numpy as np
import warnings
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy import signal
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import vozyaudio as vz
    
#########################################################################################
# Funciones de separación de canales
#########################################################################################

def extract_y(input_img):
    """
    Extrae la matriz de brillo de la imagen entrante.
    Devolverá 1 matriz de mismo ancho y alto que la de entrada.
    """
    y = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    return y

def extract_rgb(input_img):
    """
    Recibe una imagen y la separa en sus 3 canales RGB.
    Devolverá 3 matrices de mismo ancho y alto que la de entrada.
    """
    img = cv2.imread(input_img,cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)
    return r, g, b

def extract_ycbcr(input_img):
    """
    Recibe una imagen y la separa en sus 3 canales YcBcR.
    Devolverá 3 matrices de mismo ancho y alto que la de entrada.
    """
    BGRImage = cv2.imread(input_img)
    YCrCbImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(YCrCbImage)
    return y, cb, cr

#########################################################################################
# Funciones para extraccion de parametros de una imagen
#########################################################################################

def histograma(input_img, bins=256, rango=(0, 256)):
    """
    Recibe una imagen con 1 o mas canales de color
    y devuelve un array con los histogramas correspondientes
    """
    histogramas = []
    
    if len(input_img.shape) == 2: # Solo un canal de color
        hist = cv2.calcHist([input_img], [0], None, [bins], rango)
        histogramas.append(hist)
    else:                         # Mas de un canal de color
        num_canales = input_img.shape[2]
        for canal in range(num_canales):
            hist = cv2.calcHist([input_img], [canal], None, [bins], rango)
            histogramas.append(hist)
    
    return histogramas

def gradiente(imagen_gris):
    """
    Recibe una imagen de un canal y devuelve
    su gradiente en x, en y, el módulo y
    la dirección de este
    """
    
    kernel_x = np.array([ # Mascara de Prewitt para el eje x
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])

    kernel_y = np.array([ # Mascara de Prewitt para el eje y
        [-1,-1,-1],
        [ 0, 0, 0],
        [ 1, 1, 1]])
    
    grad_x = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)          # Gradiente en x
    grad_y = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)          # Gradiente en y
    magnitud = np.sqrt(grad_x**2 + grad_y**2)                         # Magnitud del gradiente
    magnitud = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX) # Normalizada para que pueda ser tratada como una imagen
    direccion = np.arctan2(grad_y, grad_x)                            # Dirección del gradiente
    return grad_x, grad_y, magnitud, direccion

def entropia(input_img):
    """
    Recibe una imagen y calcula su entropía por canal
    """
    # Asegurarnos de que es imagen en escala de grises
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Calcular histograma normalizado
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    hist_norm = hist / np.sum(hist)

    # Evitar log(0) usando una máscara
    hist_norm = hist_norm[hist_norm > 0]

    # Calcular entropía
    entropia = -np.sum(hist_norm * np.log2(hist_norm))
    return entropia

def LPB():
    """
    Recibe una imagen y compara cada pixel con sus 8 vecinos de alrededor,
    si es mayor que este el resultado será 1 y si no 0. Se construye asi un
    numero binario de 8 digitos por cada pixel de la imagen en cada canal,
    obteniendo una imagen en la que ves los patrones de intensidad de los
    pixeles por regiones.
    """
        # Asegurarnos de que es imagen en escala de grises
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Inicializar la imagen LBP
    lbp = np.zeros_like(imagen, dtype=np.uint8)

    # Definir los desplazamientos (dx, dy) de los 8 vecinos
    vecinos = [(-1, -1), (0, -1), (1, -1),
                (1, 0), (1, 1), (0, 1),
               (-1, 1), (-1, 0)]
    
    filas, cols = imagen.shape

    for y in range(1, filas-1):
        for x in range(1, cols-1):
            valor_centro = imagen[y, x]
            codigo = 0
            for idx, (dx, dy) in enumerate(vecinos):
                vecino = imagen[y + dy, x + dx]
                codigo |= (vecino >= valor_centro) << idx
            lbp[y, x] = codigo

    return lbp

#########################################################################################
# Funciones para plotear
#########################################################################################

def show_canales():
    return None

def show_histograma():
    return None

