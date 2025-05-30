#Colección de funciones y clases que componen el proyecto

#########################################################################################
# Paquetes
#########################################################################################

# === Built-in ===
import os
import sys
import shutil
import subprocess
import threading
import warnings
import random
import colorsys

# === NumPy y SciPy ===
import numpy as np
np.float = float  # para compatibilidad con algunas libs
from scipy import signal, ndimage, interpolate
from scipy.signal import (
    correlate, freqz, firwin, iirfilter, get_window,
    resample, lfilter, find_peaks
)
from scipy.io import wavfile
from scipy.fft import fft, ifft, fft2
from scipy.fftpack import dct, idct, dctn, idctn
from scipy.linalg import solve_toeplitz

# === Audio ===
import librosa
import librosa.display
from librosa import piptrack
import pyaudio
import ffmpeg
import soundfile as sf

# === Imágenes y vídeo ===
import cv2
import imageio.v2 as imageio

# === Visualización en notebooks ===
import matplotlib.pyplot as plt
import bqplot as bq
from IPython.display import Audio, Video, display
import IPython.display as ipd
import ipywidgets as widgets

# === Módulo propio: Voz y Audio ===
import vozyaudio as vz
from vozyaudio import lee_audio, sonido, envolvente, track_pitch, espectro


#########################################################################################
# De imagen a audio COMIENZO
#########################################################################################

"""
Este apartado pretende conglomerar todas las funciones desarrolladas en el proyecto
PixelSounds para creación de audio a partir de imágenes. 
"""


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
            bloque_fft = fft2(bloque)

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
            bloque_dct = dct(dct(bloque.T, norm='ortho').T, norm='ortho')
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


#########################################################################################
# De audio a video COMIENZO
#########################################################################################


#########################################################################################
# Funciones para extracción de características de audio a video
#########################################################################################

def color_fondo_por_centroide(X, fa):
    """ Genera un color RGB para el fondo en función del centroide espectral del frame.
    Entrada:
        X (numpy.ndarray): Módulo del espectro del frame de audio (magnitudes)
        fa (numpy.ndarray): Vector de frecuencias correspondientes al espectro
    Salida:
        fondo_color (tuple): Color RGB normalizado (0-1) para usar como fondo del frame
    """
    # Evitar errores si X está vacío
    if np.sum(X) == 0 or len(fa) != len(X):
        return (0, 0, 0.1)  # fondo oscuro por defecto

    # Centroide espectral
    centroide = np.sum(fa * X) / (np.sum(X) + 1e-9)

    # Normalizar centroide al rango 0–1 basado en un rango realista (0–4000 Hz)
    centroide_norm = np.clip(centroide / 4000, 0, 1)

    # Usar centroide como matiz, pero también afectar brillo
    h = centroide_norm                   # matiz (rojo ↔ azul)
    s = 0.9                              # saturación constante
    v = 0.3 + 0.7 * centroide_norm       # más agudo → más brillante

    fondo_color = colorsys.hsv_to_rgb(h, s, v)
    return fondo_color

def beat_frames(x, fs):
    """ Detecta los beats del audio a partir de la envolvente y devuelve sus ubicaciones en frames.
    Entrada:
        x (numpy.ndarray): Señal de audio completa
        fs (int): Frecuencia de muestreo del audio
    Salida:
        beat_frames (numpy.ndarray): Índices de frame donde se detectan beats en la señal
    """
    # Autocorrelación sobre la envolvente
    env_smooth = envolvente(x, fs=fs, tr=0.1)  # más estable
    corr_env = autocorrelacion(env_smooth)

    # Estimar el tempo global
    min_lag = int(fs / 5)    # máx 5 Hz = 300 BPM
    max_lag = int(fs / 1.5)  # mín 1.5 Hz = 90 BPM
    lag_beat = np.argmax(corr_env[min_lag:max_lag]) + min_lag
    periodo_muestras = lag_beat

    # Encontrar los picos en la envolvente
    peaks, _ = find_peaks(env_smooth, distance=periodo_muestras * 0.8)

    # Convertir los picos (en muestras) a tiempos (en segundos) y luego a frames
    beat_times = peaks / fs
    beat_frames = (beat_times * FPS).astype(int)
    return beat_frames

def es_beat(i, beat_frames, tolerancia=2):
    """ Determina si un frame está dentro de los limites de un beat detectado.
    Entrada:
        frame_index (int): Índice del frame actual en el video
        beat_frames (list of int): Lista de frames donde se detectaron beats
        tolerancia (int): Número de frames de margen alrededor de cada beat
    Salida:
        es_beat (bool): True si el frame está cerca de un beat, False en caso contrario
    """
    return any(abs(i - bf) <= tolerancia for bf in beat_frames)


def dibujar_flash(i, beats):
    """ Dibuja un flash visual en el centro del frame, usado para resaltar beats detectados.
    Entrada:
        beats (numpy.ndarray): Array con los índices de los fotogramas donde hay un beat
    Salida:
        None: El flash se dibuja directamente sobre la figura
    """
    # Halo pulsante animado en el centro tras beat
    for bf in beats:
        frames_from_beat = i - bf
        if 0 <= frames_from_beat <= 4:  # duración 4 frames
            grow = 1 - frames_from_beat / 4
            size = 2000 * grow
            alpha = 0.8 * grow
            plt.scatter(0.5, 0.5, s=size, c='magenta', alpha=alpha, edgecolors='none')


def detectar_ritmo(x_frame, fs, fmin=1.5, fmax=8):
    """ Estima el periodo rítmico de un fragmento de audio mediante autocorrelación.
    Entrada:
        x_frame (numpy.ndarray): Fragmento de señal de audio (ventana temporal)
        fs (int): Frecuencia de muestreo del audio
        fmin (float): Frecuencia mínima esperada del ritmo (en Hz)
        fmax (float): Frecuencia máxima esperada del ritmo (en Hz)
    Salida:
        periodo_seg (float): Periodo estimado del ritmo en segundos
        corr (numpy.ndarray): Autocorrelación normalizada del fragmento de audio
    """
    corr = autocorrelacion(x_frame)
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    if max_lag >= len(corr): max_lag = len(corr) - 1
    if min_lag >= max_lag: return 0.5, corr  # Valor por defecto
    pico = np.argmax(corr[min_lag:max_lag]) + min_lag
    periodo_seg = pico / fs
    return periodo_seg, corr


def dibujar_circulo_ritmico(t_actual, periodo):
    """ Dibuja un círculo que pulsa rítmicamente en el centro del frame según un periodo dado.
    Entrada:
        t_actual (float): Tiempo actual del video en segundos
        periodo (float): Periodo rítmico estimado en segundos
    Salida:
        None: El círculo se dibuja directamente sobre la figura
    """
    ritmo_osc = 0.5 * (1 + np.sin(2 * np.pi * t_actual / periodo))
    color = (ritmo_osc, 0.2, 1 - ritmo_osc)
    size = 300 * ritmo_osc + 20
    plt.scatter(0.5, 0.5, s=size, c=[color], alpha=0.3)


def autocorrelacion(x_frame):
    """ Calcula la autocorrelación normalizada de una ventana de señal de audio.
    Entrada:
        x_frame (numpy.ndarray): Fragmento de señal de audio (ventana temporal)
    Salida:
        corr_norm (numpy.ndarray): Autocorrelación normalizada desde el retardo cero hacia adelante
    """
    x_frame = x_frame - np.mean(x_frame)
    corr = correlate(x_frame, x_frame, mode='full')
    mid = len(corr) // 2
    return corr[mid:] / np.max(np.abs(corr) + 1e-9)


def dibujar_barras(X_resampled, N_BARRAS):
    """ Dibuja barras verticales que representan la energía en diferentes bandas de frecuencia.
    Entrada:
        X_resampled (numpy.ndarray): Vector con las amplitudes espectrales reescaladas a N_BARRAS bandas
        N_BARRAS (int): Número total de barras a dibujar
    Salida:
        None: Las barras se dibujan directamente sobre la figura
    """
    bar_width = 1 / N_BARRAS
    for j in range(N_BARRAS):
        height = X_resampled[j]
        color = (0.1, 0.8 * height, 1.0)
        plt.bar(j * bar_width, height, width=bar_width*0.8, color=color, align='edge')

def crear_video(audio_path, out_path, frames_dir="fotogramas", framerate=25, ffmpeg_path="ffmpeg.exe"):
    try:
        if os.path.exists(out_path):
            os.remove(out_path)

        input_video = os.path.join(frames_dir, "frame_%04d.png")

        # Construir comando subprocess
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate", str(framerate),
            "-i", input_video,
            "-i", audio_path,
            "-vcodec", "libx264",
            "-acodec", "pcm_s32le",
            "-pix_fmt", "yuv420p",
            "-shortest",
            out_path
        ]
        subprocess.run(cmd, check=True)

        print(f"\n[OK] Video exportado como: {out_path}")

    except subprocess.CalledProcessError as e:
        print("Error al crear el video:")
        print(e)

    except FileNotFoundError:
        print(f"No se encontró ffmpeg en la ruta: {ffmpeg_path}")

def dibujar_particula(pitch, env):
    """ Dibuja una partícula que se mueve según el pitch y cambia de tamaño según la envolvente del audio
    Entrada:
        pitch (float) : Valor de estimación del pitch en un fotograma determinado
        env (float) : Valor de la envolvente del audio en un fotograma determinado
    Salida:
        None: La función solo dibuja en la figura y no devuelve nada
    """
    y_pos = pitch
    size = 100 + env * 300
    color = (1.0, env, pitch)
    plt.scatter(0.5, y_pos, s=size, c=[color], alpha=0.8)

def normalizar(v):
    """ Normaliza un vector al rango [0, 1].
    Entrada:
        v (numpy.ndarray): Vector de valores (por ejemplo, envolvente, pitch, espectro, etc.)
    Salida:
        v_norm (numpy.ndarray): Vector normalizado en el rango [0, 1]
    """
    return (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-9)

def obtener_descriptores(x,fs):
    """ Extrae distintos descriptores de la señal de audio.
    Entrada:
        x (numpy.ndarray): Vector de valores de la señal
        fs (int): Frecuencia de muestreo de la señal
    Salida:
        pitch_frame (numpy.ndarray): Vector con los valores de la estimación del pitch normalizados y redimensionados al número de frames
        env_frame (numpy.ndarray): Vector con los valores de la envolvente normalizados y redimensionados al número de frames 
    """
    # Extraemos los descriptores de envolvente y estimación de pitch
    env = envolvente(x, fs=fs) # Extraer envolvente
    pitch = track_pitch(x, fs) # Estimar pitch
    pitch = np.nan_to_num(pitch)  # Reemplaza NaNs por 0
    
    env = normalizar(env) # Normalizar ambos arrays
    pitch = normalizar(pitch)
    
    # Redimensionar descriptores al número de frames
    env_frame = np.interp(np.linspace(0, len(env), n_frames), np.arange(len(env)), env)
    pitch_frame = np.interp(np.linspace(0, len(pitch), n_frames), np.arange(len(pitch)), pitch)

    return pitch_frame, env_frame

def generar_frames(x,fs, FRAME_FOLDER):
    """ Genera los frames para el vídeo y los guarda en la carpeta FRAME_FOLDER
    Entrada:
        x (numpy.ndarray): Vector de valores de la señal
        fs (int): Frecuencia de muestreo de la señal
        FRAME_FOLDER (string): Ruta de la carpeta destino
    Salida:
        None: No devuelve nada, solo genera los fotogramas      
    """
    pitch_frame, env_frame = obtener_descriptores(x,fs)
    
    print("Generando frames...")
    for i in range(n_frames):
        porcentaje = (i / (n_frames-1)) * 100
        print(f"\rCompletado {porcentaje:.2f} %", end="", flush=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        
        beats = beat_frames(x,fs)
        
        # Flash más visible en beat
        if es_beat(i,beats,2):  # mayor tolerancia
            plt.scatter(0.5, 0.5, s=1500, c='cyan', alpha=0.9, edgecolors='none', marker='o')
        
        dibujar_flash(i,beats)
            
        #  Obtener trozo de señal actual
        start = i * samples_per_frame
        end = min(len(x), start + samples_per_frame)
        x_frame = x[start:end]

        # Espectro (resample a N barras)
        X, fa = espectro(x_frame, modo=1, fs=fs)
        X_resampled = resample(X, N_BARRAS)
        X_resampled = normalizar(X_resampled)
        
        #------NUEVO------
        
        fondo_color =  color_fondo_por_centroide(X,fa)
        fig.set_facecolor(fondo_color)
        
        #------NUEVO------
        
        # Detección rítmica simple
        periodo, corr = detectar_ritmo(x_frame, fs)
        
        # Calcula un pulso visual que oscila con el ritmo detectado
        t_actual = i / FPS
        
        dibujar_circulo_ritmico(t_actual,periodo)

        # Visual: Círculo que sube/baja con pitch y cambia tamaño con envolvente
        dibujar_particula(pitch_frame[i], env_frame[i])
        
        dibujar_barras(X_resampled, N_BARRAS)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{FRAME_FOLDER}/frame_{i:04d}.png")
        plt.close(fig)
    
#########################################################################################
# Clases para codificación y decodificación de audio a video
#########################################################################################

class PixelSoundsEncoder:
    """
    Convierte un audio en un “vídeo pixelado” codificando cada bloque de
    muestras en filas de imágenes.

    - Primer frame: metadatos (N, hop, fps, color_mode).
    - Siguientes frames: bloques de audio coloreados (gris o RGB).
    - Luego empaqueta PNGs en MP4 con ffmpeg.
    """
    def __init__(
        self,
        audio_path,
        frames_dir="fotogramas",
        export_dir="exports",
        fps=60,
        color_mode="color",   # 'gris' o 'color'
        map_mode="ampl",       # 'ampl', 'fft' o 'fir'
        window_type="hann",
        numcoef=101
    ):
        self.audio_path  = audio_path
        self.frames_dir  = frames_dir
        self.export_dir  = export_dir
        self.fps         = fps
        self.color_mode  = color_mode
        self.map_mode    = map_mode
        self.window_type = window_type
        self.iscolor     = 1 if color_mode == "color" else 0

        # Leer y normalizar audio de 16-bit
        self.fs, audio = lee_audio(audio_path)
        self.audio     = audio.astype(np.float32)
        self.audio    /= np.max(np.abs(self.audio)) + 1e-12

        # Diseñar filtros FIR
        self.b_low  = firwin(numcoef,       cutoff=3000,                   fs=self.fs)
        self.b_band = firwin(numcoef, [3000,10000], pass_zero=False,         fs=self.fs)
        self.b_high = firwin(numcoef,       cutoff=10000, pass_zero=False,   fs=self.fs)

        # Usar ffmpeg local (Windows -  como los ordenadores del aula)
        self.ffmpeg_path = os.path.abspath("ffmpeg.exe")
        if not os.path.isfile(self.ffmpeg_path):
            raise FileNotFoundError("No se encontró 'ffmpeg.exe' en el directorio del proyecto.")

        # Crear carpetas de salida
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def _write_header(self, N, hop):
        """
        Escribe el frame 0000 como cabecera con metadatos de N, hop, fps y color_mode.
        """
        # crear imagen vacia NxN
        header = np.zeros((N, N), dtype=np.uint8)

        # codificar N y hop en bytes alto y bajo
        hiN, loN = (N>>8)&0xFF, N&0xFF
        hiH, loH = (hop>>8)&0xFF, hop&0xFF

        # rellenar filas con metadatos
        header[0,:] = hiN
        header[1,:] = loN
        header[2,:] = hiH
        header[3,:] = loH
        header[4,:] = self.fps & 0xFF         # fps
        header[5,:] = self.iscolor            # flag color

        # guardar como frame_0000
        path = os.path.join(self.frames_dir, "frame_0000.png")
        imageio.imwrite(path, header)

    def _colorear_fila(self, fila, modo):
        """
        Aplica mapeado de color a la fila segun el modo.

        - ampl → R = amp, G = 255-amp, B = constante
        - fft  → R = mag, G = phase, B = constante
        - fir  → R/G/B = bandas low/band/high

        Devuelve un array Nx3 uint8 (RGB por pixel).
        """

        if modo == 'ampl':
            # si es color, cogemos solo el canal rojo
            amp = fila[:,0] if fila.ndim == 2 else fila

            # rojo: valor original
            r = amp
            # verde: complementario para dar contraste
            g = 255 - amp
            # azul: constante (centro)
            b = 128 * np.ones_like(amp, dtype=np.uint8)

        elif modo == 'fft':
            # usamos magnitud (R) y fase (G), B fijo
            mag_n, phase_n = fila[:,0], fila[:,1]
            r = mag_n
            g = phase_n
            b = 255 * np.ones_like(r, dtype=np.uint8)  # canal azul a tope

        elif modo == 'fir':
            # ya vienen como low/band/high → R/G/B
            r, g, b = fila[:,0], fila[:,1], fila[:,2]

        else:
            raise ValueError(f"Modo desconocido para colorear: {modo}")

        # ensamblar los tres canales en un solo array (Nx3)
        return np.stack([r, g, b], axis=1).astype(np.uint8)

    def generate_frames(self):
        """
        Divide el audio en bloques, aplica codificacion y guarda cada uno como PNG.
        """
        # 1) Calcular tamaño de bloque y hop
        N   = self.fs // self.fps
        if N % 2: N += 1               # aseguramos par
        hop = N // 2

        # 2) Guardar cabecera
        self._write_header(N, hop)

        # 3) Ventana y num de bloques
        window   = get_window(self.window_type, N, fftbins=True)
        n_blocks = (len(self.audio) - N) // hop
        print(f"[Encoder] N={N}, HOP={hop}, FPS={self.fps}, Bloques={n_blocks}")

        # 4) Procesar bloque a bloque
        for i in range(n_blocks):
            start = i * hop
            block = self.audio[start:start+N] * window

            # 4.1) Codificar segun map_mode
            if self.map_mode == "ampl":
                # normalizar el bloque a [0,1] y escalar a 8 bits
                norm = (block - block.min()) / (block.max() - block.min() + 1e-12)
                fila = (norm * 255).astype(np.uint8)  # fila resultante, uint8

            elif self.map_mode == "fft":
                # calcular la FFT
                X       = fft(block, n=N)
                X       = np.fft.fftshift(X)          # centrar la FFT
                mag     = np.abs(X)                   # magnitud
                phase   = np.angle(X)                 # fase en radianes

                # normalizar magnitud a [0,255]
                mag_n   = np.round((mag / (mag.max() + 1e-12)) * 255).astype(np.uint8)
                # convertir fase de [-pi, pi] → [0, 255]
                phase_n = np.round(((phase + np.pi) / (2*np.pi)) * 255).astype(np.uint8)

                # construir fila RGB: R=mag, G=phase, B=0
                fila    = np.stack([mag_n, phase_n, np.zeros_like(mag_n)], axis=1)

            elif self.map_mode == "fir":
                # aplicar 3 filtros FIR: low, band y high
                y_l = lfilter(self.b_low,  1.0, block)
                y_b = lfilter(self.b_band, 1.0, block)
                y_h = lfilter(self.b_high, 1.0, block)

                # recortar cada señal a [-1,1]
                y_l, y_b, y_h = map(lambda y: np.clip(y, -1, 1), (y_l, y_b, y_h))

                # convertir a int8 → uint8 (para guardar como imagen)
                r8 = np.round(y_l * 127).astype(np.int8).view(np.uint8)
                g8 = np.round(y_b * 127).astype(np.int8).view(np.uint8)
                b8 = np.round(y_h * 127).astype(np.int8).view(np.uint8)

                # construir fila RGB
                fila = np.stack([r8, g8, b8], axis=1)

            else:
                raise ValueError("map_mode debe ser 'ampl', 'fft' o 'fir'")


            # 4.2) Construir imagen segun color_mode
            if self.color_mode == "gris":
                if self.map_mode == "fft":
                    # intercalar mag y phase
                    img = np.empty((N, N), dtype=np.uint8)
                    img[0::2, :] = mag_n[np.newaxis, :]
                    img[1::2, :] = phase_n[np.newaxis, :]
                elif self.map_mode == "fir":
                    # intercalar bandas y repetir verticalmente
                    img = np.empty((N, N), dtype=np.uint8)
                    img[0::3, :] = r8[np.newaxis, :]
                    img[1::3, :] = g8[np.newaxis, :]
                    img[2::3, :] = b8[np.newaxis, :]
                else:
                    gris = fila if fila.ndim == 1 else fila[:,0]
                    img  = np.tile(gris[np.newaxis,:], (N,1))
            else:
                # modo color
                base_rgb = fila if fila.ndim == 2 else np.stack([fila]*3, axis=1)
                colored  = self._colorear_fila(base_rgb, self.map_mode)
                img      = np.tile(colored[np.newaxis,...], (N,1,1))

            # 4.3) Guardar imagen PNG
            path = os.path.join(self.frames_dir, f"frame_{i+1:04d}.png")
            imageio.imwrite(path, img)

        print(f"[Encoder] Frames generados en '{self.frames_dir}/'")


    def encode_video(self, output_name=None):
        """
        Empaqueta los frames PNG como un MP4 sin pérdida y añade el audio reconstruido si está disponible.

        - Video: libx264, CRF 0, YUV444p
        - Audio (opcional): reconstruido desde generate_frames, codificado como AAC
        """
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        name = output_name or f"{base}_{self.map_mode}_{self.color_mode}.mp4"
        out_video = os.path.join(self.export_dir, name)

        # Eliminar si ya existe
        if os.path.exists(out_video):
            os.remove(out_video)

        # Paso 1: Generar MP4 a partir de imágenes
        cmd = [
            self.ffmpeg_path, "-y",
            "-framerate", str(self.fps),
            "-i", os.path.join(self.frames_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
            out_video
        ]

        subprocess.run(cmd, check=True)
        print(f"[Encoder] Video sin audio exportado en '{out_video}'")
        return out_video

class PixelSoundsDecoder:
    """
    Decodifica un vídeo generado por PixelSoundsEncoder de vuelta a WAV.

    - Primer frame: metadatos (N, hop, fps, color_mode).
    - Siguientes frames: bloques de audio (gris o RGB) codificados.
    - Reconstruye por overlap-add y guarda WAV.
    """
    def __init__(
        self,
        frames_dir,
        output_wav,
        map_mode='ampl',    # 'ampl', 'fft' o 'fir'
        window_type='hann'  # tipo de ventana para overlap-add
    ):
        self.frames_dir  = frames_dir
        self.output_wav  = output_wav
        self.map_mode    = map_mode
        self.window_type = window_type

    def extract_all_frames(self, video_path, prefix="frame_", fmt="png"):
        """
        Extrae todos los frames de un vídeo y los guarda como imágenes PNG numeradas
        en la carpeta self.frames_dir. Limpia el contenido previo si existe.

        Parámetros:
        - video_path: ruta al vídeo del que extraer los frames
        - prefix: prefijo para nombrar los archivos generados
        - fmt: formato de imagen de salida (por defecto 'png')
        """
        print(f"[Decoder] Extrayendo frames de '{video_path}'...")

        # si existe la carpeta, la limpiamos por completo
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)

        # abrimos el vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el vídeo {video_path}")

        # bucle para leer y guardar todos los frames
        idx = 0
        while True:
            ret, frame = cap.read()           # intentamos leer el siguiente frame
            if not ret:
                break                         # fin del vídeo
            fname = f"{prefix}{idx:04d}.{fmt}"  # frame_0000.png, frame_0001.png, ...
            cv2.imwrite(os.path.join(self.frames_dir, fname), frame)
            idx += 1

        cap.release()
        print(f"[Decoder] Total frames extraídos: {idx}")
        return idx


    def extraer_metadatos_cabecera_rows(self, frame):
        """Lee la cabecera y extrae N, hop, fps y si está en color"""
        print("[Decoder] Leyendo metadatos de header...")
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if frame.ndim == 3:
            frame = frame[..., 0]  # usar solo canal rojo si es RGB

        # extraer bytes altos y bajos para N y hop
        hiN, loN = int(frame[0,0]), int(frame[1,0])
        hiH, loH = int(frame[2,0]), int(frame[3,0])
        fps      = int(frame[4,0])
        flag     = int(frame[5,0])

        N        = (hiN << 8) | loN
        hop      = (hiH << 8) | loH
        is_color = bool(flag)

        print(f"[Decoder] Header -> N={N}, hop={hop}, fps={fps}, modo={'color' if is_color else 'gris'}")
        return N, hop, fps, is_color

    def decode(self):
        """
        Reconstruye el audio a partir de los PNG ya extraídos en self.frames_dir.
        Asume que frame_0000.png (header) y los frames de datos están presentes.
        """
        # 1) Leer metadatos desde el header
        header = imageio.imread(os.path.join(self.frames_dir, "frame_0000.png"))
        N, hop, fps, self.is_color = self.extraer_metadatos_cabecera_rows(header)
        fs_recon = N * fps

        # 2) Obtener listado de frames ignorando el header
        files   = sorted(f for f in os.listdir(self.frames_dir)
                        if f.startswith("frame_") and f != "frame_0000.png")
        n_blocks = len(files)

        # 3) Inicializar buffers para reconstrucción por superposición (overlap-add)
        length  = N + hop * (n_blocks - 1)
        audio   = np.zeros(length, dtype=np.float32)
        pesos   = np.zeros(length, dtype=np.float32)
        ventana = get_window(self.window_type, N, fftbins=True)

        # 4) Decodificar bloque a bloque
        for i, fname in enumerate(files, start=1):
            path = os.path.join(self.frames_dir, fname)
            raw  = imageio.imread(path)  # imagen uint8

            if self.is_color:
                # === Modo color ===
                row_uint8 = raw[0]  # usamos solo la primera fila

                if self.map_mode == 'ampl':
                    amp_norm = row_uint8[:,0].astype(np.float32) / 255.0
                    block    = amp_norm * 2.0 - 1.0

                elif self.map_mode == 'fft':
                    mag_n    = row_uint8[:,0].astype(np.float32) / 255.0
                    phase_n  = (row_uint8[:,1].astype(np.float32) / 255.0) * 2*np.pi - np.pi
                    X        = mag_n * np.exp(1j * phase_n)
                    X        = np.fft.ifftshift(X)
                    block    = np.real(ifft(X, n=N))

                elif self.map_mode == 'fir':
                    # cada canal representa una banda: low, band, high
                    pix_c = np.ascontiguousarray(row_uint8)
                    y_l   = pix_c[:,0].view(np.int8).astype(np.float32) / 127.0
                    y_b   = pix_c[:,1].view(np.int8).astype(np.float32) / 127.0
                    y_h   = pix_c[:,2].view(np.int8).astype(np.float32) / 127.0
                    block = y_l + y_b + y_h

                else:
                    raise ValueError(f"map_mode desconocido: {self.map_mode}")

            else:
                # === Modo gris ===
                gray = raw[...,0] if raw.ndim == 3 else raw

                if self.map_mode == 'ampl':
                    pix   = gray[0].astype(np.float32)
                    block = (pix / 255.0) * 2.0 - 1.0

                elif self.map_mode == 'fft':
                    mag_n    = gray[0, :].astype(np.float32) / 255.0
                    phase_n  = (gray[1, :].astype(np.float32) / 255.0) * 2*np.pi - np.pi
                    X        = mag_n * np.exp(1j * phase_n)
                    X        = np.fft.ifftshift(X)
                    block    = np.real(ifft(X, n=N))

                elif self.map_mode == 'fir':
                    # filas intercaladas 0::3,1::3,2::3
                    y_l = gray[0::3, :][0].view(np.int8).astype(np.float32) / 127.0
                    y_b = gray[1::3, :][0].view(np.int8).astype(np.float32) / 127.0
                    y_h = gray[2::3, :][0].view(np.int8).astype(np.float32) / 127.0
                    # recombinar bandas
                    block = y_l + y_b + y_h

                else:
                    raise ValueError(f"map_mode desconocido: {self.map_mode}")

            # 4.2) Superposición por ventana
            start = (i-1) * hop
            audio[start:start+N] += block * ventana
            pesos[start:start+N] += ventana

        # 5) Normalizar por ventana y guardar
        audio /= (pesos + 1e-12)
        sonido(audio, fs_recon)  # opcional: reproducir
        os.makedirs(os.path.dirname(self.output_wav) or '.', exist_ok=True)
        scaled = np.int16(np.clip(audio, -1, 1) * 32767)
        wavfile.write(self.output_wav, fs_recon, scaled)

        return audio, fs_recon


#########################################################################################
# De audio a video FIN
#########################################################################################
