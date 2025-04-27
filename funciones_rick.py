import vozyaudio as vz
import numpy as np
import matplotlib.pyplot as plt
import vozyaudio as vz
import subprocess
import os
import shutil


def generarFrame(fa, espectro, energia, centroide, i, tamBloq, url_audio):
    """
    Genera un fotograma RGB basado en el centroide y la energia del bloque.

    fa: frecuencias del espectro
    espectro: magnitud espectral
    energia: energia temporal del bloque
    centroide: centroide espectral
    i: indice del fotograma
    tamBloq: tamaño del fotograma (pixeles)
    url_audio: ruta al audio (no usado)
    """
    fotograma = np.zeros((tamBloq, tamBloq, 3), dtype=np.float64)

    centroide_norm = centroide / (np.max(fa) + 1e-12)

    if centroide_norm < 0.33:
        color = np.array([0, 0, 1])  # azul
    elif centroide_norm < 0.66:
        color = np.array([1, 1, 0])  # amarillo
    else:
        color = np.array([1, 0, 0])  # rojo

    brillo = energia / (energia + 1e-12)
    brillo = np.clip(energia, 0, 1)
    fotograma[:] = brillo * color

    plt.imsave(f'fotogramas/bloque_{i:04d}.png', fotograma)


def calcular_descriptores(bloque, fs):
    """
    Calcula descriptores espectrales y energéticos de un bloque de audio.

    Parámetros:
    bloque : ndarray
        Bloque de muestras de audio (1D).
    fs : int o float
        Frecuencia de muestreo en Hz.

    Devuelve:
    espectro : ndarray
        Espectro de magnitud del bloque.
    fa : ndarray
        Vector de frecuencias asociado al espectro.
    energia : float
        Energía temporal del bloque (suma de muestras al cuadrado).
    centroide : float
        Centroide espectral (frecuencia promedio ponderada por energía).
    entropia : float
        Entropía espectral (medida de aleatoriedad o dispersión).
    freq_90 : float
        Frecuencia hasta la cual se acumula el 90% de la energía espectral.
    """

    # Espectro
    espectro, fa = vz.espectro(bloque, fs=fs)
    
    # Energia del bloque
    energia = np.sum(bloque**2)

    # Centroide espectral
    centroide = np.sum(fa * espectro) / (np.sum(espectro) + 1e-12)

    # Entropia espectral
    prob = espectro / (np.sum(espectro) + 1e-12)
    entropia = -np.sum(prob * np.log2(prob + 1e-12))

    # Frecuencia máxima significativa (90% energía)
    energia_total = np.sum(espectro)
    energia_acumulada = np.cumsum(espectro) 
    indice_max_freq = np.searchsorted(energia_acumulada, 0.9 * energia_total)
    freq_90 = fa[min(indice_max_freq, len(fa) - 1)]  # por si acaso

    return espectro, fa, energia, centroide, entropia, freq_90


def ricardo(audio, fs, url_audio):
    """
    Procesa un audio por bloques, genera fotogramas y guarda descriptores.

    audio: array de la señal de audio
    fs: frecuencia de muestreo
    url_audio: ruta al archivo de audio (para el posterior montaje de video)
    """
    
    # Eliminar y crear carpeta de fotogramas
    if os.path.exists('fotogramas'):
        shutil.rmtree('fotogramas')
    os.makedirs('fotogramas')
    
    # Definición de tamaño de bloque
    tamBloq = fs // 25  # 25 FPS
    salto = tamBloq     # No solape
    print(f"TamBloq {tamBloq}, Seg {fs/tamBloq:.2f}, Fs {fs}")
    
    # Número total de bloques
    n_bloques = int(np.floor((len(audio) - tamBloq) / salto))

    # Procesado por bloques
    for i in range(n_bloques):
        porcentaje = (i / n_bloques) * 100
        print(f"\rCompletado {porcentaje:.2f} %", end="", flush=True)
        
        # Obtener bloque (sin ventana)
        bloque = audio[i * salto : i * salto + tamBloq]

        # Analizar bloque
        espectro, fa, energia, centroide, entropia, freq_90 = calcular_descriptores(bloque, fs)

        # Generar fotograma
        generarFrame(fa, espectro, energia, centroide, i, tamBloq, url_audio)
    
    print("\nFotogramas generados.")

def generarVideo(url_audio):
    try:
        subprocess.run(['generarVideo.bat', url_audio], check=True)
    except subprocess.CalledProcessError as e:
        print("Error al ejecutar generarVideo.bat:", e)

    finally:
        shutil.rmtree('fotogramas/')

        print('Procesado terminado.')