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

def extract_rgb(input_img):
    """
    Esta función debe recibir una imagen y separarla en sus 3 canales RGB.
    Devolverá 3 matrices de mismo ancho y alto que la de entrada.
    """
    img = cv2.imread(input_img,cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)
    return r, g, b

def extract_y(input_img):
    """
    Extrae la matriz de brillo de la imagen entrante.
    Devolverá 1 MATRIZ de mismo ancho y alto que la de entrada.
    """
    y = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    return y

def extract_ycbcr(input_img):
    """
    Esta función debe recibir una imagen y separarla en sus 3 canales YcBcR.
    Devolverá 3 matrices de mismo ancho y alto que la de entrada.
    """
    BGRImage = cv2.imread(input_img)
    YCrCbImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(YCrCbImage)
    return y, cb, cr

#########################################################################################
# Funciones para extraccion de parametros de una imagen
#########################################################################################

def histograma(input_img):
    """
    Recibe una imagen con 1 o más canales y devuelve el histograma de esta
    Por hacer
    """
    
    return None

def gradiente(imagen_gris,tipo="X"):

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [-1,-1,-1],
        [ 0, 0, 0],
        [ 1, 1, 1]
    ])
    
    grad_x = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)

    grad_y = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)

    # Magnitud del gradiente
    magnitud = np.sqrt(grad_x**2 + grad_y**2)
    magnitud = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)

    # Dirección del gradiente
    direccion = np.arctan2(grad_y, grad_x)

    return grad_x, grad_y, magnitud, direccion

#########################################################################################
# Funciones para plotear
#########################################################################################

def show_canales():
    return None

def show_histograma():
    return None

