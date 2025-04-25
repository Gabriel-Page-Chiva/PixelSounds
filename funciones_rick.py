import numpy as np
import matplotlib.pyplot as plt
import vozyaudio as vz
import subprocess
import os
import shutil


def generarVideo(fa, espectro, energia, centroide, i, tamBloq, url_audio):
    # Crear fotograma RGB
    fotograma = np.zeros((tamBloq, tamBloq, 3), dtype=np.float64)

    # Normalizar espectro y centroide para visualizar
    espectro_norm = espectro / (np.max(espectro) + 1e-12)
    idx_frec = (fa / np.max(fa) * (tamBloq - 1)).astype(int)
    fotograma[:, idx_frec, 0] = espectro_norm  # canal R para espectro

    # Dibujar energía (barra vertical en el borde izquierdo)
    energia_norm = energia / (np.max(espectro) * tamBloq)  # normalización relativa
    alto = int(energia_norm * tamBloq)
    fotograma[tamBloq - alto:, 0:10, 1] = 1  # canal G para energía

    # Dibujar centroide (línea horizontal)
    centroide_idx = int(centroide / (np.max(fa) + 1e-12) * tamBloq)
    if 0 <= centroide_idx < tamBloq:
        fotograma[centroide_idx, :, 2] = 1  # canal B para centroide

    # Guardar imagen
    plt.imsave(f'fotogramas/bloque_{i:04d}.png', fotograma)

    # Si es el ultimo bloque, generar video
    if i == -1:
        subprocess.run([
            'ffmpeg', '-framerate', '25',
            '-i', 'fotogramas/bloque_%04d.png',
            '-i', f'{url_audio}',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            'salida.mp4'
        ])
        shutil.rmtree('fotogramas')

def calcular_descriptores(bloque, fs):
    espectro, fa = vz.espectro(bloque, fs=fs)
    
    # Energia del bloque
    energia = np.sum(bloque**2)

    # Centroide espectral: indica hacia dónde está concentrada la energía en frecuencia
    # Valores altos: sonido más brillante / agudo; valores bajos: sonido más grave
    centroide = np.sum(fa * espectro) / (np.sum(espectro) + 1e-12)

    return espectro, fa, energia, centroide

def ricardo(audio, fs, url_audio, tamBloq=1024, salto=512, ventana='hann'):
    """
    Procesa un audio por bloques, genera fotogramas y guarda descriptores.

    audio: array de la señal de audio
    fs: frecuencia de muestreo
    tamBloq: Tamaño de bloque
    salto: Salto entre bloques
    ventana: Tipo de ventana (por defecto 'hann')
    """
    # Crear carpeta de salida si no existe
    os.makedirs('fotogramas', exist_ok=True)

    # Procesado por bloques
    n_bloques = int(np.floor((len(audio) - tamBloq) / salto))

    descriptores = {
        'fs': fs,
        'tamBloq': tamBloq,
        'salto': salto,
        'n_bloques': n_bloques,
        'energia': [],
        'centroide': []
    }

    for i in range(n_bloques):
        # Obtener bloque
        bloque = audio[i*salto : i*salto + tamBloq]
        bloque = bloque * np.hanning(tamBloq)
        print(bloque,fs)

        # Analizar bloque
        espectro, fa, energia, centroide = calcular_descriptores(bloque, fs)

        # Guardar descriptores
        descriptores['energia'].append(energia)
        descriptores['centroide'].append(centroide)

        # Generar fotograma
        generarVideo(fa, espectro, energia, centroide, i, tamBloq, url_audio)

    print('Procesado terminado.')

    # Guardar descriptores para revertir
    np.save('descriptores.npy', descriptores)