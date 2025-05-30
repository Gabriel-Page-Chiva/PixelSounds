#########################################################################################
# De imagen a audio COMIENZO
#########################################################################################

"""
Este apartado pretende conglomerar todas las funciones de momento desarrolladas en el proyecto
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
    Calcula el histograma normalizado de una imagen en escala de grises.
    
    Parámetros:
    - img_gray: imagen en escala de grises (matriz 2D).
    - bins: número de contenedores del histograma.
    - rango: rango de valores de los píxeles.

    Devuelve:
    - hist: array 1D con los valores del histograma.
    """
    hist = cv2.calcHist([imagen_gris], [0], None, [bins], rango)
    hist /= np.sum(hist)
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
    magnitud = np.ones(magnitud.shape) - magnitud
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

#########################################################################################
# Funciones para creación de audio
#########################################################################################

def frecuencia_a_midi(freqs):
    """
    Convierte una lista de frecuencias a notas MIDI enteras y las restringe al rango 60-96

    Args:
        freqs (numpy.ndarray): Array de frecuencias.

    Returns:
        numpy.ndarray: Notas MIDI
    """
    notas_midi = np.round(69 + 12 * np.log2(freqs / 440)).astype(int)
    return notas_midi

def imagen_a_audio_dft(imagen_gris, fs, duracion_nota=0.5, duracion_total=10, num_frecuencias=75, tam_bloque=8, dur=np.array([False]), amp=np.array([False]), desv=np.array([False])):
    """
    Convierte una imagen en escala de grises en un audio utilizando la FFT 2D
    por bloques. Selecciona las frecuencias visuales dominantes y las convierte en tonos audibles.

    Args:
        imagen_gris (np.ndarray): Imagen 2D en escala de grises.
        fs (int): Frecuencia de muestreo del audio.
        duracion_nota (float): Duración de cada nota en segundos.
        duracion_total (float): Duración total del audio en segundos.
        num_frecuencias (int): Número de frecuencias a extraer.
        tam_bloque (int): Tamaño de los bloques cuadrados para FFT.
        dur (np.ndarray or None): Duraciones personalizadas.
        amp (np.ndarray or None): Amplitudes personalizadas.
        desv (np.ndarray or None): Desviaciones para modulación.

    Returns:
        np.ndarray: Señal de audio generada.
    """
    # Redimensionar imagen
    imagen = cv2.resize(imagen_gris, (256, 256))
    alto, ancho = imagen.shape
    coeficientes = []
    
    # FFT 2D por bloques
    for i in range(0, alto, tam_bloque):
        for j in range(0, ancho, tam_bloque):
            bloque = imagen[i:i+tam_bloque, j:j+tam_bloque].astype(float)
            bloque_fft = fft.fft2(bloque)

            # Extraer magnitudes, ignorando componente DC
            mag = np.abs(bloque_fft)
            mag[0, 0] = 0  # Ignora DC
            mitad = tam_bloque // 2
            coeficientes.extend(mag[:mitad, :mitad].flatten())

    coeficientes = np.array(coeficientes)

    # Seleccionar los índices con mayor magnitud
    indices_principales = np.argsort(coeficientes)[-num_frecuencias:]

    # Mapear esos índices a frecuencias arbitrarias dentro del rango audible
    freqs = np.logspace(np.log10(20), np.log10(fs/2), len(coeficientes))  # Espacio de frecuencias
    frecuencias_principales = freqs[indices_principales]

    # Convertir a notas MIDI
    notas_midi = frecuencia_a_midi(frecuencias_principales)

    # Recortar al número de notas posible según la duración
    num_notas = min(len(notas_midi), int(duracion_total / duracion_nota))
    
    notas_midi = notas_midi[:num_notas]
    if dur.all() == False:
        dur = np.full(num_notas, duracion_nota)
    if amp.all() == False:
        amp = np.ones_like(notas_midi)
    if desv.all() == False:
        desv = np.zeros_like(notas_midi)

    # Generar audio
    audio, tiempo = vz.generar_tono_pitchmidi(notas_midi, dur, amp, desv, fs)
    return audio

def imagen_a_audio_dct(imagen_gris, fs, duracion_nota=0.5, duracion_total=10, num_frecuencias=75, tam_bloque=8, dur=np.array([False]), amp=np.array([False]), desv=np.array([False])):
    """
    Procesa una imagen en escala de grises, aplica la DCT por bloques y genera una señal de audio a partir
    de las frecuencias principales codificadas en los coeficientes DCT.

    Args:
        imagen_gris (np.ndarray): Imagen de un solo canal (grises).
        fs (int): Frecuencia de muestreo para el audio.
        duracion_nota (float): Duración de cada nota en segundos.
        duracion_total (float): Duración total del audio en segundos.
        num_frecuencias (int): Número de frecuencias principales a extraer.
        tam_bloque (int): Tamaño de bloque para la DCT (por defecto 8x8).

    Returns:
        np.ndarray: Señal de audio generada.
    """

    # Redimensionar imagen
    imagen = cv2.resize(imagen_gris, (256, 256))
    alto, ancho = imagen.shape
    coeficientes = []

    # Aplicar DCT 2D por bloques
    for i in range(0, alto, tam_bloque):
        for j in range(0, ancho, tam_bloque):
            bloque = imagen[i:i+tam_bloque, j:j+tam_bloque].astype(float)
            bloque_dct = fft.dct(fft.dct(bloque.T, norm='ortho').T, norm='ortho')
            # Aplanar y guardar los coeficientes (ignorando el DC [0,0])
            coeficientes.extend(np.abs(bloque_dct.flatten()[1:]))

    coeficientes = np.array(coeficientes)

    # Seleccionar los índices con mayor magnitud
    indices_principales = np.argsort(coeficientes)[-num_frecuencias:]

    # Mapear esos índices a frecuencias arbitrarias dentro del rango audible
    freqs = np.logspace(np.log10(20), np.log10(fs/2), len(coeficientes))  # Espacio de frecuencias
    frecuencias_principales = freqs[indices_principales]

    # Convertir a notas MIDI
    notas_midi = frecuencia_a_midi(frecuencias_principales)

    # Recortar al número de notas posible según la duración
    num_notas = min(len(notas_midi), int(duracion_total / duracion_nota))
    
    notas_midi = notas_midi[:num_notas]
    if dur.all() == False:
        dur = np.full(num_notas, duracion_nota)
    if amp.all() == False:
        amp = np.ones_like(notas_midi)
    if desv.all() == False:
        desv = np.zeros_like(notas_midi)

    # Generar audio
    audio, tiempo = vz.generar_tono_pitchmidi(notas_midi, dur, amp, desv, fs)
    return audio

#########################################################################################
# Funciones para mapeo de características de imagen en audio
#########################################################################################

def mapeo_color(r, g, b, fs, duracion=10):
    """
    Mapea los canales R, G y B a desviación, amplitud y duración.
    """
    n_notas = fs*duracion
    
    # Promedios globales de cada canal
    r_mean = np.mean(r, axis=1)
    g_mean = np.mean(g, axis=1)
    b_mean = np.mean(b, axis=1)

    # Normaliza a [0,1]
    r_norm = r_mean / 255
    g_norm = g_mean / 255
    b_norm = b_mean / 255

    # Mapea a pitch MIDI entre 40 y 100 (E2 a E7)
    desv = 40 + r_norm * 60
    desv = desv.astype(int)

    # Mapea amplitud entre 0.3 y 1.0
    amp = 0.3 + g_norm * 0.7

    # Mapea duración entre 0.2 y 1.5s
    dur = 0.2 + b_norm * 1.3

    # Ajustar el tamaño de los arrays
    desv = np.tile(desv,int(np.ceil(n_notas/len(desv))))
    desv = desv[:n_notas]

    amp = np.tile(amp,int(np.ceil(n_notas/len(amp))))
    amp = amp[:n_notas]

    dur = np.tile(dur,int(np.ceil(n_notas/len(dur))))
    dur = dur[:n_notas]
    
    return dur, amp, desv


def mapeo_histograma(histograma, x, fs=16000):
    """
    A partir del valor medio de la derivada del
    histograma de una imagen, aplica unnvibrato 
    a una señal de audio.
    """
    hist_norm     = histograma/np.max(histograma)
    hist_norm     = np.squeeze(hist_norm)
    der_hist_norm = np.diff(hist_norm)
    fm            = 10*np.mean(der_hist_norm)
    Afm           = np.mean(der_hist_norm)/10
    vibrato       = vz.vibrato(x,5, 0.01, fs)
    return vibrato

def mapeo_gradiente(gradiente,aplicar_ventana=True):
    """
    Función de mapeo que genera coeficientes de un filtro 
    FIR a partir de la media de las filas de una imagen de 
    gradiente.

    Parámetros:
    - gradiente: array 2D, imagen con la magnitud del gradiente (normalmente positiva)
    - aplicar_ventana: si se aplica ventana de Hamming para suavizar los extremos

    Devuelve:
    - coef: array 1D de coeficientes FIR normalizados
    """
    # Paso 1: Normalizar la imagen (evita distorsión por rango alto de valores)
    grad_norm = gradiente.astype(np.float32)
    grad_norm /= grad_norm.max() if grad_norm.max() > 0 else 1

    # Paso 2: Calcular el perfil medio por fila
    coef = np.mean(grad_norm, axis=1)  # 1 valor por fila

    # Paso 4: Aplicar ventana de Hamming si se desea
    if aplicar_ventana:
        ventana = signal.get_window("hamming",len(coef))
        coef = coef*ventana

    # Paso 5: Normalizar energía total del filtro (evita amplificación o atenuación global)
    coef /= np.sum(coef) if np.sum(coef) != 0 else 1
    return coef

def mapeo_LBP(LBP):
    """
    Función de mapeo que calcula la
    entropía de una imagen LBP y 
    devuelve una cantidad de bits con
    los que se recuantificará el audio.
    """
    entr = entropia(LBP)
    bits = np.floor(entr)
    return bits

#########################################################################################
# De imagen a audio FIN
#########################################################################################