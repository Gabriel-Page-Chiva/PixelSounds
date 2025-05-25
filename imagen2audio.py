"""
Este archivo pretende conglomerar todas las funciones de momento desarrolladas en el proyecto
PixelSounds para creación de audio a partir de imágenes. 
"""

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
import random
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, median_filter, gaussian_filter
from scipy import signal
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import vozyaudio as vz
np.float = float  

#########################################################################################
# Funciones para extracción de características de imágenes
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
    su gradiente en valor absoluto
    """
    
    kernel_x = np.array([ # Mascara de Prewitt para el eje x
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])

    kernel_y = np.array([ # Mascara de Prewitt para el eje y
        [-1,-1,-1],
        [ 0, 0, 0],
        [ 1, 1, 1]])
    
    grad_x   = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)        # Gradiente en x
    grad_y   = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)        # Gradiente en y
    magnitud = np.sqrt(grad_x**2 + grad_y**2)                         # Magnitud del gradiente
    magnitud = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX) # Normalizada para que pueda ser tratada como una imagen
    return magnitud

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


"""
def LBP(imagen_gris):
    
    Calcula el Local Binary Pattern (LBP) de una imagen en escala de grises (2D) de forma vectorizada.

    Parámetros:
    - imagen_gris: imagen 2D en escala de grises.

    Devuelve:
    - lbp: imagen 2D con los valores LBP.
    

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

    return lbp"""
    
def LBP(imagen_gris):
    imagen_gris = imagen_gris.astype(np.uint8)
    
    # Definimos los 8 vecinos (dy, dx) y sus pesos
    offsets = [(-1, -1), (0, -1), (1, -1),
               (1,  0), (1,  1), (0,  1),
               (-1, 1), (-1, 0)]
    pesos = [1 << i for i in range(8)]

    # Preparamos la salida y la región central
    h, w = imagen_gris.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    centro = imagen_gris[1:h-1, 1:w-1]
    
    # Altura y anchura de la región válida
    hc, wc = centro.shape

    for (dy, dx), peso in zip(offsets, pesos):
        # Extraemos una ventana EXACTAMENTE del mismo tamaño que 'centro',
        # desplazada (dy, dx)
        y0, x0 = 1 + dy, 1 + dx
        vecino = imagen_gris[y0 : y0 + hc,
                             x0 : x0 + wc]
        # Ahora sí coincide: both vecino and centro are (hc, wc)
        lbp[1:h-1, 1:w-1] |= ((vecino >= centro) * peso).astype(np.uint8)

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

#########################################################################################
# Funciones para creación de audio
#########################################################################################

def frecuencia_a_midi(freqs):
    """
    Convierte una lista de frecuencias a notas MIDI enteras y las restringe al rango 60-96

    Args:
        freqs (numpy.ndarray): Array de frecuencias.

    Returns:
        numpy.ndarray: Notas MIDI restringidas al rango 48-96 correspondiente a las notas C2 - C7
    """
    
    # https://newt.phys.unsw.edu.au/jw/notes.html
    notas_midi = np.round(69 + 12 * np.log2(freqs / 440)).astype(int) # De frecuencia a nota MIDI es 69 + 12 * log2(f / 440)
    return np.clip(notas_midi, 48, 96)

def nota_midi_a_frecuencia(nota_midi):
    return 440.0 * (2 ** ((nota_midi - 69) / 12.0))

# genera las frecuencias de una escala musical a partir de una nota raiz, tipo de escala y rango de octavas
def generar_frecuencias_escala(nota_raiz=None, tipo_escala=None, octava_base=3, num_octavas=3):
    nombres_notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    intervalos_mayor = [0, 2, 4, 5, 7, 9, 11]  # Intervalos para la escala mayor
    intervalos_menor = [0, 2, 3, 5, 7, 8, 10]  # Intervalos para la escala menor

    # selecciona una nota raiz y tipo de escala aleatoriamente si no se proporcionan
    if nota_raiz is None:
        nota_raiz = random.choice(nombres_notas)
    if tipo_escala is None:
        tipo_escala = random.choice(['mayor', 'menor'])

    indice_raiz = nombres_notas.index(nota_raiz)
    intervalos = intervalos_mayor if tipo_escala == 'mayor' else intervalos_menor
    frecuencias_escala = []
    # genera las frecuencias para cada octava y cada intervalo de la escala
    for octava in range(octava_base, octava_base + num_octavas):
        for intervalo in intervalos:
            nota_midi = indice_raiz + intervalo + (octava * 12)
            freq = nota_midi_a_frecuencia(nota_midi)
            frecuencias_escala.append(freq)
    return np.array(frecuencias_escala)

# ajusta una frecuencia dada a la frecuencia mas cercana en una escala musical
def ajustar_a_escala(frecuencia, escala_frecuencias):
    return escala_frecuencias[np.argmin(np.abs(escala_frecuencias - frecuencia))]

# fusiona una melodia y un ritmo, mezclandolos con diferentes pesos
def fusionar_melodia_y_ritmo(melodia, ritmo):
    longitud = min(len(melodia), len(ritmo))  # asegura que ambas señales tengan la misma longitud
    combinado = melodia[:longitud] * 0.2 + ritmo[:longitud] * 0.6  # mezcla las señales con pesos
    combinado /= np.max(np.abs(combinado) + 1e-8)  # normaliza la señal combinada
    return combinado