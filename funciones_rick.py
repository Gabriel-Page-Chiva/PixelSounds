import os
import shutil
import subprocess
import numpy as np
import cv2
import imageio.v2 as imageio
from scipy.signal import firwin, lfilter, get_window
from scipy.fft import fft, ifft
from scipy.io import wavfile as wav
from vozyaudio import lee_audio, sonido

class PixelSoundsEncoder:
    """
    Convierte un audio en un “vídeo pixelado” codificando cada bloque de
    muestras en filas de imágenes.

    - Primer frame: metadatos (N, hop, fps, color_mode).
    - Siguientes frames: bloques de audio coloreados (gris o RGB).
    - Luego empaqueta PNGs en MP4 con un batch externo.
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
        Llama a ffmpeg para empaquetar los frames PNG como un MP4 sin perdida.

        - Usa libx264 en modo sin compresion (crf 0)
        - Preset "veryslow" para mejor compresion sin afectar calidad
        - Formato de color: yuv444p
        """
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        name = output_name or f"{base}_{self.map_mode}_{self.color_mode}.mp4"
        out  = os.path.join(self.export_dir, name)

        # borrar si ya existia
        if os.path.exists(out):
            os.remove(out)

        # construir comando ffmpeg
        cmd = [
            "ffmpeg",
            "-y",                            # sobrescribe sin preguntar
            "-framerate", str(self.fps),    # fps usado en codificacion
            "-i", os.path.join(self.frames_dir, "frame_%04d.png"),  # input por patron
            "-c:v", "libx264",              # codec de video
            "-crf", "0",                    # calidad sin perdida
            "-preset", "veryslow",          # compresion optima
            "-pix_fmt", "yuv444p",          # formato RGB completo
            out
        ]

        # ejecutar ffmpeg
        subprocess.run(cmd, check=True)
        print(f"[Encoder] Video exportado en '{out}'")
        return out


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
        wav.write(self.output_wav, fs_recon, scaled)

        return audio, fs_recon
