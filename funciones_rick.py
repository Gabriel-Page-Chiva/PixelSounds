import os
import shutil
import subprocess
import numpy as np
import cv2
import imageio.v2 as imageio
from scipy.signal import firwin, lfilter, get_window
from scipy.fft import fft, ifft
from scipy.io import wavfile as wav
from vozyaudio import lee_audio


class PixelSoundsEncoder:
    """
    Convierte un audio en un “vídeo pixelado” codificando cada bloque de
    muestras en filas de imágenes.

    - Primer frame: metadatos (N, hop, fps).
    - Siguientes frames: bloques de audio coloreados (gris o RGB).
    - Luego empaqueta PNGs en MP4 con un batch externo.
    """
    def __init__(
        self,
        audio_path,
        frames_dir="fotogramas",
        export_dir="exports",
        fps=60,
        color_mode="color",   # "gris" o "color"
        map_mode="ampl",      # "ampl", "fft" o "fir"
        window_type="hann",
        numcoef=101
    ):
        """
        Inicializa el codificador.

        audio_path   : ruta al WAV de entrada.
        frames_dir   : carpeta donde escribir PNGs.
        export_dir   : carpeta de salida de vídeo.
        fps          : frames por segundo del vídeo.
        color_mode   : 'gris' o 'color'.
        map_mode     : 'ampl', 'fft' o 'fir'.
        window_type  : tipo de ventana (p.ej. 'hann').
        numcoef      : número de coeficientes FIR (solo para 'fir').
        """
        self.audio_path  = audio_path
        self.frames_dir  = frames_dir
        self.export_dir  = export_dir
        self.fps         = fps
        self.color_mode  = color_mode
        self.map_mode    = map_mode
        self.window_type = window_type

        # Leer y normalizar audio
        self.fs, audio = lee_audio(audio_path)
        self.audio     = audio.astype(np.float32)
        self.audio    /= np.max(np.abs(self.audio)) + 1e-12

        # Filtros FIR
        self.b_low  = firwin(numcoef, cutoff=2000,                   fs=self.fs)
        self.b_band = firwin(numcoef, [2000, 6000], pass_zero=False, fs=self.fs)
        self.b_high = firwin(numcoef, cutoff=6000, pass_zero=False,  fs=self.fs)

        # Preparar carpetas
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def _write_header(self, N, hop):
        header = np.zeros((N, N), dtype=np.uint8)
        hiN, loN = (N >> 8) & 0xFF, N & 0xFF
        hiH, loH = (hop >> 8) & 0xFF, hop & 0xFF
        b_fps     = self.fps & 0xFF

        header[0, :] = hiN
        header[1, :] = loN
        header[2, :] = hiH
        header[3, :] = loH
        header[4, :] = b_fps

        path = os.path.join(self.frames_dir, "frame_0000.png")
        imageio.imwrite(path, header)

    def _color_row_reversible(self, block, N):
        mode = self.map_mode

        if mode == "ampl":
            X24 = np.round((block + 1) / 2 * (2**24 - 1)).astype(np.uint32)
            hi = (X24 >> 16) & 0xFF
            md = (X24 >>  8) & 0xFF
            lo =  X24        & 0xFF
            row = np.stack([hi, md, lo], axis=1)[np.newaxis, ...]

        elif mode == "fft":
            X     = fft(block, n=N)
            mag   = np.abs(X)
            phase = np.angle(X)
            mag_n   = mag   / (mag.max() + 1e-12)
            phase_n = (phase + np.pi) / (2*np.pi)
            r = np.round(mag_n   * 255).astype(np.uint8)
            g = np.round(phase_n * 255).astype(np.uint8)
            b = np.zeros_like(r, dtype=np.uint8)
            row = np.stack([r, g, b], axis=1)[np.newaxis, ...]

        elif mode == "fir":
            y_l = lfilter(self.b_low,  1.0, block)
            y_b = lfilter(self.b_band, 1.0, block)
            y_h = lfilter(self.b_high,1.0, block)
            y_l, y_b, y_h = map(lambda y: np.clip(y, -1, 1), (y_l,y_b,y_h))
            r8 = np.round(y_l *127).astype(np.int8)
            g8 = np.round(y_h *127).astype(np.int8)
            b8 = np.round(y_b *127).astype(np.int8)
            r = r8.view(np.uint8)
            g = g8.view(np.uint8)
            b = b8.view(np.uint8)
            row = np.stack([r, g, b], axis=1)[np.newaxis, ...]

        else:
            raise ValueError("map_mode debe ser 'ampl', 'fft' o 'fir'")

        return row

    def generate_frames(self):
        N = self.fs // self.fps
        if N % 2:
            N += 1
        hop = N // 2
        self._write_header(N, hop)
        window = get_window(self.window_type, N)
        n_blocks = (len(self.audio) - N) // hop

        for i in range(n_blocks):
            start = i * hop
            block = self.audio[start:start+N] * window
            if self.color_mode == "gris":
                norm = (block - block.min()) / (block.max() - block.min() + 1e-12)
                row  = (norm * 255).astype(np.uint8)
                img  = np.tile(row, (N, 1))
            else:
                row = self._color_row_reversible(block, N)
                img = np.tile(row, (N, 1, 1))
            path = os.path.join(self.frames_dir, f"frame_{i+1:04d}.png")
            imageio.imwrite(path, img)

    def encode_video(self, output_name=None):
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        name = output_name or f"{base}_{self.map_mode}_{self.color_mode}.mp4"
        out  = os.path.join(self.export_dir, name)
        if os.path.exists(out):
            os.remove(out)
        subprocess.run(
            ["cmd", "/c", "generarVideo2.bat", str(self.fps), out],
            check=True
        )
        return out


class PixelSoundsDecoder:
    """
    Decodifica un vídeo generado por PixelSoundsEncoder de vuelta a WAV.
    """
    def __init__(
        self,
        frames_dir,
        output_wav,
        map_mode='ampl',
        window_type='hann'
    ):
        self.frames_dir  = frames_dir
        self.output_wav  = output_wav
        self.map_mode    = map_mode
        self.window_type = window_type

    def _read_header(self):
        header = imageio.imread(os.path.join(self.frames_dir, "frame_0000.png"))
        if header.ndim == 3:
            header = header[..., 0]
        hiN = int(header[0, 0]); loN = int(header[1, 0])
        hiH = int(header[2, 0]); loH = int(header[3, 0])
        fps = int(header[4, 0])
        N   = (hiN << 8) | loN
        hop = (hiH << 8) | loH
        return N, hop, fps

    def extract_all_frames(self, video_path, prefix="frame_", fmt="png"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el vídeo {video_path}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fname = f"{prefix}{idx:04d}.{fmt}"
            path = os.path.join(self.frames_dir, fname)
            cv2.imwrite(path, frame)
            idx += 1
        cap.release()
        return idx

    def extraer_metadatos_cabecera_rows(self, frame):
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if frame.ndim == 3:
            frame = frame[..., 0]
        hiN = int(frame[0, 0]); loN = int(frame[1, 0])
        hiS = int(frame[2, 0]); loS = int(frame[3, 0])
        fps = int(frame[4, 0])
        N   = (hiN << 8) | loN
        hop = (hiS << 8) | loS
        return N, hop, fps

    def decode(self, video_path):
        # 1) Cabecera
        N, hop, fps = self._read_header()
        # 2) Extraer frames
        n = self.extract_all_frames(video_path)
        # 3) Listar frames
        files = sorted(f for f in os.listdir(self.frames_dir)
                       if f.startswith("frame_") and f != "frame_0000.png")
        n_blocks = len(files)
        length   = N + hop * (n_blocks - 1)
        audio    = np.zeros(length, dtype=np.float32)
        window   = get_window(self.window_type, N)
        # 4) Procesar
        for i, fname in enumerate(files):
            img = imageio.imread(os.path.join(self.frames_dir, fname))
            pix = img[0]
            if self.map_mode == 'ampl':
                hi = pix[:, 0].astype(np.uint32)
                md = pix[:, 1].astype(np.uint32)
                lo = pix[:, 2].astype(np.uint32)
                X24 = (hi << 16) | (md << 8) | lo
                block = (X24 / (2**24 - 1)) * 2 - 1
            elif self.map_mode == 'fft':
                mag_n   = pix[:, 0].astype(np.float32) / 255.0
                phase_n = pix[:, 1].astype(np.float32) / 255.0
                mag     = mag_n
                phase   = phase_n * 2*np.pi - np.pi
                X       = mag * np.exp(1j * phase)
                block   = np.real(ifft(X, n=N))
            elif self.map_mode == 'fir':
                r8 = pix[:, 0].view(np.int8).astype(np.float32) / 127.0
                block = r8
            else:
                raise ValueError("map_mode debe ser 'ampl', 'fft' o 'fir'")
            start = i * hop
            audio[start:start+N] += block * window

        # Comprobar si existe la carpeta decoded

        # 5) Guardar WAV
        scaled = np.int16(np.clip(audio / (np.max(np.abs(audio))+1e-12), -1, 1) * 32767)
        wav.write(self.output_wav, fps * N // N, scaled)
        print(f"WAV reconstruido en: {self.output_wav}")
        return audio, fps



if __name__ == "__main__":
    # 1) Rutas y parámetros
    video_path  = "exports/music_ampl_color.mp4"  # tu MP4 generado por el encoder
    frames_dir  = "sintesis"                      # carpeta donde volcar los PNG
    output_wav  = "decoded/mi_audio_recon.wav"      # WAV de salida
    map_mode    = "ampl"                            # debe coincidir con el encoder
    window_type = "hann"

    # # 2) Instanciar el decoder
    # decoder = PixelSoundsDecoder(
    #     frames_dir=frames_dir,
    #     output_wav=output_wav,
    #     map_mode=map_mode,
    #     window_type=window_type
    # )
    # print("1")
    # # 3) Extraer todos los fotogramas del vídeo
    # total = decoder.extract_all_frames(video_path, prefix="frame_", fmt="png")
    # print(f"[Decoder] {total} fotogramas extraídos en '{frames_dir}/'")

    # print("1")
    # # 4) Decodificar la señal aprovechando los fotogramas recién extraídos
    # audio, fs = decoder.decode(video_path)
    # print(f"[Decoder] Audio reconstruido ({fs} Hz) y guardado en '{output_wav}'")
    # print("1")




    # TESTS

    def extraer_metadatos_cabecera_rows(frame):
        """
        Extrae N, salto y fps de un array 2D o 3D codificado por filas:
        fila 0→hiN, 1→loN, 2→hiS, 3→loS, 4→fps.
        """
        # asegurar uint8 y escala de grises
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if frame.ndim == 3:
            frame = frame[..., 0]

        hiN  = int(frame[0, 0]);     loN  = int(frame[1, 0])
        hiS  = int(frame[2, 0]);     loS  = int(frame[3, 0])
        fpsb = int(frame[4, 0])

        N     = (hiN << 8) + loN
        salto = (hiS << 8) + loS
        fps   = fpsb

        return N, salto, fps


    ruta_frame = os.path.join("fotogramas", "frame_0000.png")
    frame_orig = imageio.imread(ruta_frame)

    # Extrae y muestra los metadatos
    N, salto, fps = extraer_metadatos_cabecera_rows(frame_orig)
    print(f"Original → N: {N}, salto: {salto}, fps: {fps}")
    ruta_frame = os.path.join("fotogramas", "frame_0000.png")
    frame_orig = imageio.imread(ruta_frame)

    # Extrae y muestra los metadatos
    N, salto, fps = extraer_metadatos_cabecera_rows(frame_orig)
    print(f"Original → N: {N}, salto: {salto}, fps: {fps}")