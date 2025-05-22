#########################################################################################
# Paquetes
#########################################################################################

import cv2
import os
import numpy as np
import librosa
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

def histograma(imagen_gris, bins=256, rango=(0, 256)):
    """
    Calcula el histograma de una imagen en escala de grises.
    
    Parámetros:
    - img_gray: imagen en escala de grises (matriz 2D).
    - bins: número de contenedores del histograma.
    - rango: rango de valores de los píxeles.

    Devuelve:
    - hist: array 1D con los valores del histograma.
    """
    hist = cv2.calcHist([imagen_gris], [0], None, [bins], rango)
    return hist

def gradiente(imagen_gris):
    """
    Recibe una imagen de un canal y devuelve
    su gradiente en x, en y y el módulo de este
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
    return grad_x, grad_y, magnitud

def entropia(imagen_gris):
    """
    Calcula la entropía de una imagen en escala de grises (2D).
    
    Parámetros:
    - imagen_gris: imagen 2D en escala de grises.

    Devuelve:
    - entropía (float): medida de la cantidad de información en la imagen.
    """

    hist         = histograma(imagen_gris)                       # Histograma 
    hist_norm    = hist/ hist.sum()                              # Histograma normalizado
    hist_nonzero = hist_norm[hist_norm > 0]                      # Evitar log(0)
    entropia     = -np.sum(hist_nonzero * np.log2(hist_nonzero)) # Calcular entropía
    return entropia

def LBP(imagen_gris):
    """
    Calcula el Local Binary Pattern (LBP) de una imagen en escala de grises (2D) de forma vectorizada.

    Parámetros:
    - imagen_gris: imagen 2D en escala de grises.

    Devuelve:
    - lbp: imagen 2D con los valores LBP.
    """

    imagen_gris = imagen_gris.astype(np.uint8)          # Asegurar que la imagen es tipo uint8
    offsets     = [(-1, -1), (0, -1), (1, -1),          # Definir desplazamientos y pesos en sentido horario
                   (1,  0), (1,  1), (0,  1),
                   (-1, 1), (-1, 0)]
    
    pesos  = [1 << i for i in range(8)]                 # [1, 2, 4, 8, 16, 32, 64, 128]
    lbp    = np.zeros_like(imagen_gris, dtype=np.uint8) # Imagen LBP
    centro = imagen_gris[1:-1, 1:-1]                    # Imagen sin bordes (zona válida)

    for (dy, dx), peso in zip(offsets, pesos):
        vecino = imagen_gris[1+dy: -1+dy, 1+dx: -1+dx]
        lbp[1:-1, 1:-1] |= ((vecino >= centro) * peso).astype(np.uint8)

    return lbp


def DFT_img(imagen_gris):
    """
    Calcula la Transformada Discreta de Fourier (DFT) 2D de una imagen en escala de grises,
    y devuelve los promedios de magnitud espectral a lo largo de los ejes X e Y.
    """
    
    img = imagen_gris.astype(np.float32) # Asegurar tipo float para precisión
    DFT = fft.fft2(img)                  # Aplicar FFT 2D
    F_shift = fft.fftshift(DFT)
    modulo = np.abs(F_shift)             # Modulo del espectro
    DFT_x = np.mean(modulo, axis=0)      # Media de las frecuencias de las columnas
    DFT_y = np.mean(modulo, axis=1)      # Media de las frecuencias de las filas
    return DFT_x, DFT_y

def DCT_img(imagen_gris):
    """
    Calcula la Transformada Discreta del Coseno (DCT) 2D de una imagen en escala de grises,
    y devuelve los promedios de magnitud espectral a lo largo de los ejes X e Y.
    """
    
    img = imagen_gris.astype(np.float32)                        # Asegurar tipo float para precisión
    DCT = fft.dct(fft.dct(img.T, norm='ortho').T, norm='ortho') # DCT por filas, luego por columnas (2D)
    magnitud = np.abs(DCT)                                      # Modulo de la transformada
    DCT_x = np.mean(magnitud, axis=0)                           # Media de las frecuencias de las columnas
    DCT_y = np.mean(magnitud, axis=1)                           # Media de las frecuencias de las filas
    return DCT_x, DCT_y



#########################################################################################
# Creación de audio a partir de características de imágenes
#########################################################################################

def espectrograma_por_bloques(imagen_gris, block_size=16, return_db=False):
    """
    Divide la imagen en bloques horizontales (como ventanas STFT)
    y calcula el espectro 1D (por filas) de cada bloque.
    
    Devuelve un espectrograma simulado (2D: frecuencia vs tiempo).
    """
    h, w          = imagen_gris.shape              # Filas, columnas de la imagen
    n_blocks      = w // block_size                # cantidad de bloques horizontales
    img           = imagen_gris.astype(np.float32)
    espectrograma = []

    for i in range(n_blocks):
        inicio = i * block_size
        fin = inicio + block_size
        bloque   = img[:, inicio:fin]          # Bloque vertical: todas las filas, block_size columnas
        señal    = bloque.mean(axis=1)         # Promediamos para tener una señal 1D (como una ventana de audio)
        espectro = np.abs(np.fft.fft(señal))   # FFT del bloque
        espectro = espectro[:len(espectro)//2] # Espectro unilateral
        espectrograma.append(espectro)

    espectrograma = np.array(espectrograma).T # Convertimos lista a matriz (freq x tiempo simulado)
    
    if return_db:
        espectrograma_db = 20 * np.log10(espectrograma + 1e-5) # Escala logarítmica (opcional)
        return espectrograma_db
    else:
        return espectrograma

def flatness(espectrograma):
    """
    Calcula la flatness de un espectrograma dado
    Parámetros:
    - espectrograma: ndarray 2D; un espectrograma
    
    Output:
    - flatness: ndarray 1D; un vector que contiene las
    medidas de flatness del espectrograma dado.
    """
    flatness = librosa.feature.spectral_flatness(S=espectrograma)
    return flatness

def centroide(espectrograma):
    """
    Calcula el centroide de un espectrograma dado
    Parámetros:
    - espectrograma: ndarray 2D; un espectrograma
    
    Output:
    - centroide: ndarray 1D; un vector que contiene las
    medidas del centroide espectral del espectrograma dado.
    """
    centroide = librosa.feature.spectral_centroid(S=espectrograma)
    return centroide

def ancho_banda(espectrograma):
    """
    Calcula el ancho de banda espectral de un espectrograma dado
    Parámetros:
    - espectrograma: ndarray 2D; un espectrograma
    
    Output:
    - ancho_banda: ndarray 1D; un vector que contiene las
    medidas del ancho de banda espectral del espectrograma dado.
    """
    ancho_banda = librosa.feature.spectral_bandwidth(S=espectrograma)
    return ancho_banda

def imagen_a_audio(espectrograma, sr=22050, dur_total=2.0):
    """
    Convierte un espectrograma simulado (freq x time_blocks) en audio.
    Cada bloque se convierte en una suma de senoidales según su espectro.
    """
    n_freqs, n_blocks = espectrograma.shape
    dur_bloque = dur_total / n_blocks
    t_bloque = np.linspace(0, dur_bloque, int(sr * dur_bloque), endpoint=False)
    audio_total = np.zeros(int(sr * dur_total))

    freqs = np.linspace(0, sr // 2, n_freqs)  # frecuencias correspondientes a cada bin FFT

    for i in range(n_blocks):
        amplitudes = espectrograma[:, i]
        amplitudes = amplitudes / (np.max(amplitudes) + 1e-6)  # normalización
        bloque_audio = np.zeros_like(t_bloque)

        for f, amp in zip(freqs, amplitudes):
            bloque_audio += amp * np.sin(2 * np.pi * f * t_bloque)

        # insertamos el bloque en la posición correspondiente
        inicio = int(i * dur_bloque * sr)
        fin = inicio + len(t_bloque)
        audio_total[inicio:fin] += bloque_audio

    # normalizamos audio final
    audio_total = audio_total / np.max(np.abs(audio_total) + 1e-6)

    return audio_total
