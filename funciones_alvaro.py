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

#########################################################################################
# Funciones para plotear
#########################################################################################

def show_canales():
    return None

def show_histograma():
    return None

