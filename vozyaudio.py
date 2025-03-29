"""Funciones y utilidades para la realización de las prácticas de la asignatura Voz y Audio Digital."""
import numpy as np 
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy.io.wavfile as wav
from scipy.signal import correlate, freqz, firwin, iirfilter, get_window, resample, lfilter
from scipy.fftpack import dct, idct, dctn, idctn
from scipy.linalg import solve_toeplitz
from librosa import piptrack
import ipywidgets as widgets
import threading
import bqplot as bq
import pyaudio
import librosa
import librosa.display
from scipy import signal
from scipy import ndimage
from scipy import interpolate
import sys


grabacion_mic=[]

def midi2f(c_MIDI):
    """Devuelve la frecuencia correspondiente con el código c_MIDI introducido como argumento.
     Argumentos de entrada:
         c_MIDI (int): Entero con el código MIDI del sonido buscado

    Salida:
         f (float): frecuencia del sonido buscado

    """ 
    f=440*2**((c_MIDI-69)/12)
    return f


def f2midi(f):
    """Devuelve el código correspondiente con la frecuencia introducida como argumento y su desviación en céntimos de nota.
     Argumentos de entrada:
         f (float): frecuencia del sonido 
         
    Salida:
        pitch (int): código MIDI de la frecuencia más cercana a la introducida
        desv (float): desviación de la nota representada con la frecuencia introducida respecto de la representada por el código MIDI devuelto
         """ 
    pitch=69+(12)*np.log2(f/440)
    return int(round(pitch)), 100*(pitch-round(pitch))

def generar_tono_pitchmidi(pitches=[69], dur=[0.5], amp=[1], desv=[0], Fs=4000):
    """Generación de melodias a partir de eventos asociando tonos al listado de Números de Notas MIDI introducido como parámetro de entrada

      Argumentos de entrada:
        pitches (list): Lista de número de notas MIDI (valor por defecto = [69])
        dur (list): Duración (en segundos) de cada evento (Valor por defecto = [0.5])
        amp (list): Amplitud de cada tono generado (valor por defecto = [1])
        desv(list): Desviación en céntimos de cada tono respecto de su valor nominal (valor por defecto [0])
        Fs (escalar): Frecuencia de muestreo (Valor por defecto = 4000)
     
     Salida:
        x (np.ndarray): Señal generada
        t (np.ndarray): Eje temporal (en segundos)
    """
    
    x = []
    t =  []
    
    for p, n, a, c in zip(pitches, dur, amp, desv):
        N = int(n* Fs)   
        t0 = np.arange(N) / Fs
        freq = (2 ** ((p - 69) / 12) * 440) * 2 ** (c/1200)
        
        x = np.append(x, a*np.sin(2 * np.pi * freq * t0))
        t =  np.append(t, t0)
        
    x =  x / np.max(x)   # Normalizamos la salida
    return x, t


def cuantif_uniforme(x,B,A,tipo='mt'):
    """Cuantifica la señal x con un cuantificador uniforme de medio paso (si tipo=='mt') o de media subida (si tipo=='mr' ) de B bits y rango dinámico entre -A y A
    Argumentos de entrada:
        x (np.ndarray): muestras de la señal a cuantificar
        B (int): número de bits del cuantificador
        A (float): Valor máximo del rango dinámico (simétrico) del cuantificador
        
    Salida:
        xq (np.ndarray): muestras cuantificadas 
    """
    if tipo=='mt':
        Delta=2*A/(2**B)
        k=np.rint(x/Delta)
        y=Delta*k
        xq=np.clip(y,(-(2**B-1)/2)*Delta-Delta/2,((2**B-1)/2)*Delta-Delta/2)
    else:
        Delta=2*A/(2**B-1)
        k=np.floor(x/Delta)
        y=Delta*k+Delta/2
        xq=np.clip(y,-A,A)
    
    
    return xq

def cuantifi_dither_noiseshaping(x,Delta,B,A,tipo='mt',modo=0):
    """Cuantifica añadiendo Dither y Noise Shaping usando un filtro de primer orden.
     Argumentos de Entrada:
        x (np.ndarray): variable con las muestras de la señal de audio cuantificada
        Delta (float): paso del cuantificador
        B (int): número de bits del cuantificador
        A (float): Valor máximo del rango dinámico (simétrico) del cuantificador
        tipo (str): 'mt'-> cuantificador de medio paso. 'mr'-> cuantificador de media subida. 
        modo (int): indica el tipo de función de densidad de probabilidad del ruido (0-> uniforme, 1-> Gaussiana)  
        
    Salida:
        out (np.ndarray): variable con las muestras de la señal de audio cuantificada con Dither""" 
    
    n=len(x);
    out=np.zeros(n)
    if modo==0:
        dith=(np.random.rand(n)-0.5)*Delta
    else:
        dith=(Delta/2)*np.random.randn(n)
        
    ep=0  
    for t in range(n):
        x_in = x[t] # Muestra actual de la señal a cuantificar
        d_in = dith[t] # Muestra actual del ruido de Dither a añadir
        u = x_in-ep
        y = u + d_in
        out[t] = cuantif_uniforme(y,B,A,tipo=tipo)
        ep = out[t]-u
    
    return out

def espectro(x, modo=1, NFFT=None, fs=None):
    """Calcula el módulo de la fft de x y la normaliza por el número de elementos de x. Devuelve dichos valores y los valores de las frecuencias en los que los ha calculado. Si modo = 1 (por defecto), solo parte unilateral. NFFT -> Número de puntos de la fft (bilateral) (por defecto igual al tamaño de x), fs -> Frecuencia de muestreo (1 por defecto = frecuencias discretas )
    
     Argumentos de Entrada:
        x (np.ndarray): variable con la forma de onda de la señal a analizar
        modo (int): 1, por defecto para indicar que devuelva sólo el espectro unilateral (0<= f <=fs/2 ). En cualquier otro caso devuelve el espectro bilateral (0<= f <=fs )
        NFFT (int): Variable para indicar el número de puntos a emplear en el cálculo de la fft. Por defecto, igual al número de elementos del x
        fs (escalar): Frecuencia de muestreo (Valor por defecto = 1, devuelve frecuencias discretas)
    
    Salida:
        X (np.ndarray): Módulo del espectro calculado
        fa (np.ndarray): Valores frecuenciales en los que se ha calculado el espectro
    
    """ 
    # Estimamos la raiz cuadrada de la densidad espectral de potencia como si la señal estuviera formada por componentes tonales
    
    # Número de puntos por defecto de la FFT igual a la longitud de la señal
    if NFFT==None:
        NFFT = len(x) 
    
    # fs por defecto ==1 (frecuencias discretas)
    if fs==None:
        fs = 1
           
    # Si NFFT es menor que la longitud de x, avisamos que estamos considerando sólo un trozo de la señal.
    if len(x)>NFFT:
        print('La señal se ha recortado a sus ', NFFT, 'primeras muestras')

    # Calculamos la FFT
    X = np.abs(np.fft.fft(x.T,NFFT))/np.min([len(x),NFFT]) 
    X=X.T
        
     # Generamos el eje de frecuencias
    
    fa = fs*np.arange(0, NFFT,dtype=np.float64)/NFFT
    if modo==1:
        X = X[0:NFFT//2+1]
        X[1:]=2*X[1:]
        fa = fa[0:NFFT//2+1]
   
            
    return X, fa



def sonido(x,fs,m='none'):
    """Abre un interfaz para la reproducción de auido
    Argumentos de Entrada:
        x (np.ndarray): variable con la forma de onda de la señal a reproducir
        fs (float): frecuencia de muestreo de la señal a reproducir
        m (variable): Por defecto vacía, y el audio se reproducirá normalizado respecto su máximo valor absoluto. En otro caso, indica el reproductor que normalice el audio respecto del formato int16
    """
    if m!='none':
        x=x.astype(np.float64)
        x=x/2**15
        ipd.display(ipd.Audio(x.T,rate=fs,normalize=False))
    else:
        ipd.display(ipd.Audio(x.T,rate=fs))
    
def lee_audio(fichero):
    """Lee fichero de audio wav (devuelve frecuencia de muestreo y array)
    Argumentos de Entrada:
        fichero (String): Cadena con la rura y el archivo de audio en formato wav
        
    Salida:
        fs (float): frecuencia de muestreo del audio
        x (np.ndarray): variable con las muestras de la señal de audio
    """    
    fs,x = wav.read(fichero)
    return fs,x


def dither(x,Delta,modo=0):
    """Añade ruido dither a la señal definida en 'x' para cuantificarla posteriormente con un cuantificador de paso 'Delta'. Si modo=0 ruido uniforme y si modo es 1 ruido Gaussiano.
     Argumentos de Entrada:
        x (np.ndarray): variable con las muestras de la señal de audio cuantificada
        Delta (float): paso del cuantificador
        modo (int): indica el tipo de función de densidad de probabilidad del ruido (0-> uniforme, 1-> Gaussiana)  
        
    Salida:
        out (np.ndarray): variable con las muestras de la señal de audio cuantificada con Dither""" 
    
    n=len(x);
    if modo==0:
        dith=(np.random.rand(n)-0.5)*Delta;
    else:
        dith=(Delta/2)*np.random.randn(n);
    out=x+dith
    return out

def leyA(x,A=87.6):
    """Aplica la transformación (de compresión-expansión) Ley A a la señal x
    
    Argumentos de Entrada:
        x (np.ndarray): variable con las muestras de la señal de audio 
        A (float): factor de expansión de la ley A
        
    Salida:
        out (np.ndarray): variable con las muestras de la señal de audio cuantificada aplicando ley A""" 
    
    x=x.astype(np.float64)
    x =  np.clip(x, -1, 1)
    Fx=np.zeros(x.shape,dtype=np.float64)
    Fx[np.where(abs(x)<(1/A))]=A*np.abs(x[abs(x)<(1/A)]/(1+np.log(A)))
    Fx[np.where(abs(x)>=(1/A))]=( 1+np.log( A*np.abs( x[abs(x)>=(1/A)])))/(1+np.log(A))
    
    return Fx*np.sign(x)


def inv_leyA(Fx,A=87.6):
    """Aplica la transformación inversa (de compresión-expansión) Ley A a la señal Fx
     
     Argumentos de Entrada:
        Fx (np.ndarray): variable con las muestras de la señal de audio cuantificada aplicando ley A 
        A (float): factor de expansión de la ley A
        
    Salida:
        out (np.ndarray): variable con las muestras de la señal de audio """
    
    Fx=Fx.astype(np.float64)
    Fx =  np.clip(Fx, -1, 1)
    y=np.zeros(Fx.shape,dtype=np.float64)
    
    y[np.where(abs(Fx)<(1/(1+np.log(A))))]=np.abs(Fx[abs(Fx)<(1/(1+np.log(A)))])*(1+np.log(A))/A
    y[np.where(abs(Fx)>=(1/(1+np.log(A))))]=np.exp(np.abs(Fx[abs(Fx)>=(1/(1+np.log(A)))])*(1+np.log(A))-1)/A
    return y*np.sign(Fx)

def leymu(x,mu=255):
    """Aplica la transformación (de compresión-expansión) Ley mu a la señal x
    Argumentos de Entrada:
        x (ndarray): variable con las muestras de la señal de audio 
        mu (float): factor de expansión de la ley mu
        
    Salida:
        out (np.ndarray): variable con las muestras de la señal de audio cuantificada aplicando ley mu   
   """
    x = np.clip(x, -1, 1)
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return x_mu
    
def inv_leymu(x_mu, mu=255.0):
    """Aplica la inversa de la transformación (de compresión-expansión) Ley mu a la señal x
     
     Argumentos de Entrada:
        x_mu (ndarray): variable con las muestras de la señal de audio cuantificada aplicando ley mu 
        mu (float): factor de expansión de la ley mu
        
    Salida:
        x (np.ndarray): variable con las muestras de la señal de audio"""
    x_mu = x_mu.astype(np.float64)
    x_mu = np.clip(x_mu, -1, 1)
    x=np.sign(x_mu)*((1+mu)**(np.abs(x_mu))-1)/mu
    return x
    
    
def filtroFIR_interactivo(N=101, fc=3500, fs=10000, window='boxcar'):
    """ Representa el módulo de la respuesta en frecuencia de un filtro FIR paso bajo
     
     Argumentos de Entrada:
        N(int): Número de coeficientes del filtro
        fc (float): Frecuencia de corte en Hz
        fs (float): Frecuencia de muestreo en Hz
        window (string): Cadena con el identificativo de algún tipo de ventana válido
    """
    
    plt.figure()
    b = firwin(N,fc,fs=fs,window=window)
    w, H = freqz(b, worN=2048)
    plt.plot(w*fs/(2*np.pi),20*np.log10(np.abs(H)))
    plt.ylim(-100,5)
    plt.xlabel('f(Hz)')
    plt.ylabel('dB')
    
def filtroIIR_interactivo(N=21, fc=2500, finf=2000, fsup=4000, fs=10000, btype='lowpass', ftype='cheby1', rp=3, rs=60):
    """ Representa la respuesta en frecuencia de un filtro IIR
     
     Argumentos de Entrada:
        N(int): Orden del filtro IIR
        fc (float): Frecuencia de corte en Hz
        finf (float): Frecuencia inferior en Hz
        fsup (float): Frecuencia superior en Hz
        fs (float): Frecuencia de muestreo en Hz
        btype (string): Cadena para indicar la naturaleza del filtro
        ftype (string): Cadena para indicar el tipo de polinomio de diseño
        rp (float): Rizado de la banda de paso en dB
        rs (float):  Rizado de la banda atenuada en dB
        
    """
    
    if btype=='lowpass' or btype=='highpass':
        ff=fc
    else:
        ff=[finf, fsup]

    cadenacodigo = "Codigo: b, a = signal.iirfilter({Nt},{ff_t}, btype = '{btypet}', ftype ='{ftypet}', rp ={rpt}, rs = {rst}, fs = {fst})" 
    b, a = iirfilter(N,ff,fs=fs,btype=btype, ftype=ftype, rp=rp, rs=rs)
    w, H = freqz(b,a, worN=2048)
    plt.figure(figsize=(20,5))
    plt.suptitle(cadenacodigo.format(Nt=N,ff_t=ff, btypet=btype, ftypet=ftype, rpt=rp, rst=rs, fst=fs))
    plt.subplot(121,ylim=[-100,5])
    plt.plot(w*fs/(2*np.pi),20*np.log10(np.abs(H)))
    plt.xlabel('f(Hz)')
    plt.ylabel('dB')
    plt.title('Módulo de la respuesta en frecuencia')
    
    plt.subplot(122)
    plt.plot(w*fs/(2*np.pi),(np.unwrap(np.angle(H))))
    plt.xlabel('f(Hz)')
    plt.ylabel('radianes')
    plt.title('Fase de la respuesta en frecuencia')
    
def filtro2ord_interactivo(fc=2500, fs=10000, G=20, Q=1, tipo='resonadorG',ejef_log=True):
    
    """ Representa la respuesta en frecuencia de un filtro IIR de segundo orden
     
     Argumentos de Entrada:
        
        fc (float): Frecuencia de corte o frecuencia central en Hz
        fs (float): Frecuencia de muestreo en Hz
        G (float): Ganancia de la banda modificada en dB
        Q (float): Factor de calidad 
        tipo (string): Cadena para indicar el tipo de filtro
        eje_log (bool): Variable boolena para indicar si se desea visualizar la respuesta en frecuencia en escala linea o logarítmica (por defecto)  
       """
    g=10**(G/20)
    k=np.tan(np.pi*fc/fs)
    if tipo=='resonadorG':
        a1=2*(k*k-1)/(1+(1/Q)*k+k*k)
        a2=(1-(1/Q)*k+k*k)/(1+(1/Q)*k+k*k)
        b0=(1+(g/Q)*k+k*k)/(1+(1/Q)*k+k*k)
        b1=a1
        b2=(1-(g/Q)*k+k*k)/(1+(1/Q)*k+k*k)

    elif tipo=='resonadorA':
        a1=2*(k*k-1)/(1+(g/Q)*k+k*k)
        a2=(1-(g/Q)*k+k*k)/(1+(g/Q)*k+k*k)
        b0=(1+(1/Q)*k+k*k)/(1+(g/Q)*k+k*k)
        b1=a1
        b2=(1-(1/Q)*k+k*k)/(1+(g/Q)*k+k*k)

    elif tipo=='shelvinglpc':
        a1=2*(k*k/g-1)/(1+np.sqrt(2/g)*k+k*k/g)
        a2=(1-np.sqrt(2/g)*k+k*k/g)/(1+np.sqrt(2/g)*k+k*k/g)
        b0=(1+np.sqrt(2)*k+k*k)/(g+np.sqrt(2*g)*k+k*k)
        b1=2*(k*k-1)/(g+np.sqrt(2*g)*k+k*k)
        b2=(1-np.sqrt(2)*k+k*k)/(g+np.sqrt(2*g)*k+k*k)

    elif tipo=='shelvinglpr':
        a1=2*(k*k-1)/(1+np.sqrt(2)*k+k*k)
        a2=(1-np.sqrt(2)*k+k*k)/(1+np.sqrt(2)*k+k*k)
        b0=(1+np.sqrt(2*g)*k+g*k*k)/(1+np.sqrt(2)*k+k*k)
        b1=2*(g*k*k-1)/(1+np.sqrt(2)*k+k*k)
        b2=(1-np.sqrt(2*g)*k+g*k*k)/(1+np.sqrt(2)*k+k*k)


    elif tipo=='shelvinghpc':
        a1=2*(g*k*k-1)/(1+np.sqrt(2*g)*k+g*k*k)
        a2=(1-np.sqrt(2*g)*k+g*k*k)/(1+np.sqrt(2*g)*k+g*k*k)
        b0=(1+np.sqrt(2)*k+k*k)/(1+np.sqrt(2*g)*k+g*k*k)
        b1=2*(k*k-1)/(1+np.sqrt(2*g)*k+g*k*k)
        b2=(1-np.sqrt(2)*k+k*k)/(1+np.sqrt(2*g)*k+g*k*k)

    else:
        a1=2*(k*k-1)/(1+np.sqrt(2)*k+k*k)
        a2=(1-np.sqrt(2)*k+k*k)/(1+np.sqrt(2)*k+k*k)
        b0=(g+np.sqrt(2*g)*k+k*k)/(1+np.sqrt(2)*k+k*k)
        b1=2*(k*k-g)/(1+np.sqrt(2)*k+k*k)
        b2=(g-np.sqrt(2*g)*k+k*k)/(1+np.sqrt(2)*k+k*k)
    
        
    a=[1,a1,a2]
    b=[b0,b1,b2]
 
    plt.figure()
    w, H = freqz(b, a, worN=2048)
    if ejef_log:
        plt.semilogx(w*fs/(2*np.pi),20*np.log10(np.abs(H)),base=2)
        locs, labels = plt.xticks()
        labels = locs.astype(int)
        plt.xticks(locs, labels)
        plt.xlim(64,w[-1]*fs/(2*np.pi));
    else:
        plt.plot(w*fs/(2*np.pi),20*np.log10(np.abs(H)))
            
    plt.ylim(-65,65)
    plt.xlabel('f(Hz)')
    plt.ylabel('dB')

    
def ecosimple_interactivo(N=25, G=1, A=0.5):
    """ Representa la respuesta en frecuencia del efecto de eco simple
     
     Argumentos de Entrada: 
     N (int): retardo del eco en número de muestras
     G (float): Ganancia del sonido directo en lineal 
     A (float): Ganancia del sonido retardo en lineal
    """
    
    plt.figure(figsize=(20,5))
    
    b = np.zeros(N+1)
    b[0]=G
    b[N]=A
    w, H = freqz(b)
    plt.subplot(121,ylim=[-30,30])
    plt.plot(w/(2*np.pi),20*np.log10(np.abs(H)+1e-15))
    plt.xlabel('$f_d$')
    plt.ylabel('dB')
    plt.title('Módulo de la respuesta en frecuencia')

    plt.subplot(122)
    plt.plot(w/(2*np.pi),np.unwrap(np.angle(H)))
    plt.xlabel('$f_d$')
    plt.ylabel('Radianes')
    plt.title('Fase de la respuesta en frecuencia')
    

def vibrato(x, fm, Afm, fs):
    """Función que aplica el efecto de vibrato a una señal de audio

      Argumentos de entrada:
        x (np.ndarray): array con la forma de onda del sonido al que se desea aplicar el efecto
        fm (escalar): velocidad de oscilación (o frecuencia de modulación) del eco en Hz
        Afm (escalar): profundidad de oscilación (o amplitud de modulación) en segundos
        Fs (scalar): Frecuencia de muestreo 
    Salida
        y (np.ndarray): Señal  con el efecto
    """

    N=np.rint(Afm*fs); # Retardo en muestras que se necesita para generar el Vibrato. 
    #Coincide con el número máximo de muestras para desplazarnos a la máxima profundidad del Vibrato 

    L=1+2*N;    # Máximo retardo de la línea de retardo.                
    buffer=np.zeros(int(L)+1); # reservamos un espacio de memoria suficiente para poder buscar en cada instante la muestra de la señal original con el retardo apropiado (añadimos una muestra más para poder interpolar el retardo fraccionario en el extremo del buffer)
    Fmd=fm/fs; # Frecuencia de modulación del Vibrato normalizada. 
    
    Nmuestras=np.size(x)
    y=np.zeros(Nmuestras) # Inicializamos a ceros una variable para ir almacenando los valores de salida de la señal con el efecto

    for n in range(Nmuestras): # Simulamos tiempo real procesando muestra a muestra
        xn=np.array([x[n]]) # simulamos adquirir una muestra de la señal de entrada
    
        # Procesamos en tiempo real 
        buffer=np.concatenate((xn,buffer[:-1])) # Almacenamos la señal en un buffer para poder buscar en él muestras de la señal adelantada en el tiempo y/o atrasada.
    
        Mod_vib=np.sin(2*np.pi*Fmd*n) # Valor de desplazamiento temporal instántaneo del vibrato para cada muestra n.
        Mues_vib=N+N*Mod_vib          # Valor del retardo instantaneo en muestras (no necesariamente enteras).
        NMues=np.floor(Mues_vib)      # Redondeo del retardo a nº de muestras enteras.
        frac=Mues_vib-NMues           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
        out=buffer[int(NMues)]*(1-frac)+buffer[int(NMues)+1]*(frac) #Interpolación lineal para calcular la salida suponiendo que el retardo a aplicar no sea entero.
    
        # Acabamos procesado en tiempo real
    
        y[n]=out; # Simulamos envío del resultado a la salida
    return y
    
    
def filtropromediado_interactivo(alfa):
    """ Representa la respuesta en frecuencia del filtro de promediado de primer orden
     
     Argumentos de Entrada: 
     alfa (float): Factor de olvido 
    """
    a=[1,-alfa]
    b=1-alfa
    w, H = freqz(b,a)
    plt.plot(w/(2*np.pi),20*np.log10(np.abs(H)+1e-15))
    plt.xlabel('$f_d$')
    plt.ylabel('dB')
    plt.ylim(-50,5)
    plt.title('Módulo de la respuesta en frecuencia')
    
def envolvente(x, tr=1.5 , fs=44100, ta=None, modo='peak'):
    '''devuelve la envolvente de la señal definida en x
    
    Argumentos de entrada:
    x (np.ndarray) -> Contiene la forma de onda sobre la que se calcula la envolvente
    tr (escalar) -> valor en segundos del tiempo de integración
    fs (escalar) -> frecuencia de muestreo en Hz
    ta (escalar) -> por defecto = tr. Se emplea si el tiempo de integración de ataque (en segundos) es diferente al especificado en tr
    modo (string) -> 'peak' (defecto) para usar detector de rectificación de onda completa. 'rms' para usar detector RMS
    '''
   
    x=x.astype(np.float64)
    y=np.zeros(x.shape,dtype=np.float64)
    if ta==None:
        ta=tr
    AT = 1 - np.exp(-2.2/(fs*ta))
    AR = 1 - np.exp(-2.2/(fs*tr))
    if modo=='peak':
        x=np.abs(x)
    else:
        x=np.square(x)
    pico = 0
    for n in range(len(x)):
        if x[n]>pico:
            alfa = AT
        else:
            alfa = AR
        pico=(1-alfa)*pico+alfa*x[n]
        y[n] = pico
        
    if modo != 'peak':
        y=np.sqrt(y)
    
    return y

def track_pitch(x,fs,nfft=2048,fmin=150,fmax=4000, hop=512):
    '''devuelve una estimación del pitch temporal
    
    Argumentos de entrada:
    x (np.ndarray) -> Contiene la forma de onda de la señal de audio
    nfft (int) -> número de puntos de la fft para el análisis
    fs (float) -> frecuencia de muestreo
    fmin (float) -> frecuencia mínima a considerar
    fmax (float) -> frecuencia máxima a considerar
    hop (int) -> número de muestras de salto entre análisis
    '''    
    pitches, magnitudes = piptrack(y=x, sr=fs, n_fft=nfft,fmin=100 ,fmax=fmax, hop_length=hop)
    nf,nt=np.shape(pitches)
    pitchtrack=np.zeros(nt)
    for t in range(nt):
        index=magnitudes[:,t].argmax()
        pitchtrack[t]=pitches[index,t]
    return pitchtrack



def lpc(x,P):
    '''devuelve los coeficientes del filtro de predicción lineal de orden P de una trama de datos x
    
    Argumentos de entrada:
    x (np.ndarray) -> Contiene la trama de la señal de audio
    P (int) -> orden del filtro de predicción lineal
    '''     
    x=x.astype(np.float64)
    N  = np.size(x)
    rx = correlate(x,x)/N
    rx=rx[N-1:]
    a = solve_toeplitz(rx[0:P],rx[1:P+1])
    Pe = rx[0] - np.dot(a,rx[1:P+1])
    a=np.append(1,-a)
    return a, Pe



def grabamicro(RATE=44100):
    '''Habilita un widget interactivo para la grabación de audio
    
    Argumentos de entrada:
    RATE (float) -> Frecuencia de muestreo de la grabación
    '''     
    WIDTH = 2
    CHANNELS = 1
    FORMAT=pyaudio.paInt16
    rec=[]
    Nofin=1

    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        flag= pyaudio.paContinue
        audio_data = np.frombuffer(in_data, dtype=np.int16)
    
        # procesar
        rec.append(audio_data)    
        out_data=1*audio_data
    
        return (bytes(out_data.astype(np.int16)),flag)

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                start=False,
                stream_callback=callback)


    botonDet=widgets.Button(
        description='Detener',
        disabled=True,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        )
    botonEmp=widgets.Button(
        description='Empezar',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        )
    display(widgets.HBox([botonEmp,botonDet]))


    def on_button_clicked(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True
        salida=np.array(rec).flatten()
        salida=salida-np.mean(salida)
        sonido(salida,RATE)
        global grabacion_mic
        grabacion_mic=salida


    def on_button_clicked2(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True
        
        

    botonDet.on_click(on_button_clicked)
    botonEmp.on_click(on_button_clicked2)

    
def codificadorlpc10(x,Ttrama,Tshift,fs=8000,ventana='hann',umbral=0.1,Orden=10,E_p=False):
    '''P, G, A = codificadorlpc10(x,T_trama,T_solape,ventana);
       Realizar una sencilla codificación de la señal de voz almacenada en x.
       Analiza x en tramas de T_trama segundos y para cada trama almacena los
       coeficientes del filtro LPC de orden 10 (por defecto), el Pitch y el factor de ganancia
       Las tramas que se consideran sordas tienen un Pitch=0
       
    Argumentos de entrada:
    x (np.ndarray) -> Contiene las muestras de la señal de audio a codificar
    Ttrama (float) -> Tiempo de duración de la trama de análisis
    Tshift (float) -> Tiempo entre tramas de análisis
    fs(float) -> Frecuencia de muestreo de la grabación
    ventana (string) -> cadena con algún indicador de función ventana válido
    Orden (int) -> Orden del filtro de predicción
    
    Salida:
    P(np.ndarray)-> vector con el valor del Pitch obtenido en cada trama de análisis
    G(np.ndarray) ->  vector con el valor de la ganancia obtenida en cada trama de análisis
    A(np.ndarray)->  matriz con los coeficientes LPC obtenidos en cada trama de análisis
    E_Pred(np.ndarray) -> vector con el error de predicción
    '''
    x=x.astype(np.float64)
    maximo=np.max(np.abs(x));
    Nb=int(np.ceil(Ttrama*fs)); 
    Nha=int(np.ceil(Tshift*fs));  
    
    NumTramas=int(1+np.ceil((len(x)-Nb)/Nha)); 

    s=np.zeros(int(NumTramas*Nha+Nb));
    s[0:len(x)]=1*x

    A=np.zeros((Orden+1,NumTramas)); # Para almacenar los coeficientes del filtro de cada trama
    P=np.zeros(NumTramas);  # Para almacenar el Pitch de cada trama
    G=np.zeros(NumTramas);  # Para almacenar la Ganacia de cada trama
    E_pred=np.zeros((Nha*NumTramas)); # Para almacenar el verdadero valor del error de predicción
    w=signal.get_window(ventana,int(Nb))
    zo=np.zeros(Orden)
    for trama in range(NumTramas):
        bloque=s[Nha*(trama):Nha*(trama)+Nb]
        energiamedia=np.mean(np.square(bloque))/np.mean(np.square(w))
        if energiamedia==0: # En caso de silencio
            a=np.append(1,np.zeros(Orden));
            e=0;
            ap=np.append(1,np.zeros(Orden));
            

        else:  # En caso de voz
            a,_=lpc(bloque*w,Orden)
            #Para el cálculo Pitch
            ap,_=lpc(bloque,Orden)
        A[:,trama]=a;
        
        error_pred,zo=signal.lfilter(ap,1,bloque,zi=zo)  
        E_pred[Nha*(trama):Nha*(trama)+Nha]=error_pred[0:Nha]
        e=np.mean(np.square(signal.lfilter(a,1,bloque*w)))
        
        Cx=signal.correlate(error_pred,error_pred)/len(error_pred) 
        Npitchmin=int(fs/300)
        Npitchmax=int(fs/60)
        if energiamedia==0:
            Cx[len(error_pred)-1]=1
        Cxx=Cx[len(error_pred)-1:len(error_pred)-1+Npitchmax]/Cx[len(error_pred)-1];
        Cxx[0:Npitchmin]=0;
        Amaxi=np.max(Cxx)  # Máximo de la autocorrelación (para determinar si la trama es sorda o sonora)
        Imaxi=np.argmax(Cxx); # posición del maximo de la autocorrelación
        if (Amaxi>umbral): # Si la señal es sonora.
            P[trama]=Imaxi 
            G[trama]=np.sqrt(e*Imaxi)
        else:
            G[trama]=np.sqrt(e);
    
    if E_p:
        if (ventana=='boxcar' and Nb==Nha):
            return P,G,A,E_pred
        else:
            print('La señal de error de predicción solo permite recuperar la señal de voz original con un enventanado rectangular y un tamaño de ventana igual al tamaño de salto')
            return P,G,A,E_pred
    else:
        return P,G,A    

def sintetizadorlpc10(A,P,G,t_trama,fs=8000,E_pred=np.array([None])):
    '''s = sintetizadorlpc(A,P,G,t_trama,fs);
    Función que devuelve una señal de voz sintetizada a partir del análisis de señales por tramas en las que se calcula para cada trama:
     _ Los coeficientes del filtro de análisis del filtro LPC de orden 10:
         se almacenan en cada columna de la matriz A
     _ La frecuencia del Pitch: se almacena en cada posición del vector P. 
        en caso de que la trama sea sorda, este valor será cero. 
     _ La ganacia a aplicar para sintetizar la trama: se almacena en cada posición del vector G
   
      El código devuelve la señal sintetizada de la señal de voz original que ha sido muestreada a fs Hz
       
    Argumentos de entrada:
    P(np.ndarray)-> vector con el valor del Pitch (periodo) obtenido en cada trama de análisis
    G(np.ndarray) ->  vector con el valor de la ganancia obtenida en cada trama de análisis
    A(np.ndarray)->  matriz con los coeficientes LPC obtenidos en cada trama de análisis
    t_trama -> Tiempo de salto entre tramas
    fs (float) -> frecuencia de muestreo
    E_pred(ndarray) -> Vector con la señal de error de predicción si no se desea que el decodificador la sintetice
    
    Salida:
    x (np.ndarray) -> Contiene las muestras de la señal de audio decodificada
    exc (np.ndarray) -> Contiene las muestras de la señal de excitación sintetizada en el decodificador
    '''


    Ntramas=len(G);
    muestras=int(np.ceil(t_trama*fs));

    s=np.zeros(Ntramas*muestras);  #Inicializamos la señal sintetizada a ceros
    exc=np.zeros(Ntramas*muestras);
    z=np.zeros(np.shape(A)[0]-1);                # Variable para almacenar el estado del filtro de síntesis en cada trama (mejora las transiciones entre trama)
    retardo0=0;
    # Sintetizamos trama a trama
    
    for trama in range(Ntramas):
   
        if P[trama]==0: # En caso de trama sorda 
            e=G[trama]*np.random.randn(muestras);
            retardo0=0  
            
        else:     # En caso de trama sonora
            T=P[trama]
            aux=np.zeros(muestras)
            aux[::int(T)]=1.0;
            aux=np.append(np.zeros(retardo0),aux)
            e=G[trama]*aux[0:muestras]
            retardo0=int(retardo0+((1+ muestras // T)*T)-muestras)
            
            while retardo0>=T:
                retardo0=int(retardo0-T)
               
        
        if E_pred[0]==None:
            out,z=signal.lfilter(np.array([1]),A[:,trama],e,zi=z);
        else:
            out,z=signal.lfilter(np.array([1]),A[:,trama],E_pred[muestras*(trama):muestras*(trama+1)],zi=z);
        exc[muestras*(trama):muestras*(trama+1)]= e   
        s[muestras*(trama):muestras*(trama+1)]= out;
       
    return s,exc
    

def analizavoz_interactivo(A,x,trama,Nb,Nha,fs,P,ventana):    
    '''analizavoz_interactivo(A,x,trama,Nb,Nha,fs,P,ventana)
       Utilidad para analizar los resultados de una codificación de voz LPC '''
        
    f,H=freqz(np.array([1]),A[:,trama-1],fs=fs)
    plt.figure(figsize=(10,3))
    n=np.arange(len(x))
    S, fa = espectro(x[Nha*(trama-1):Nha*(trama-1)+Nb], fs=fs)
    plt.subplot(121), plt.plot(fa,20*np.log10(S))
    plt.xlabel('f(Hz)')
    plt.ylabel('dB')
    plt.title('Espectro de la trama')
    plt.subplot(122), plt.plot(f,20*np.log10(np.abs(H)))
    plt.xlabel('f(Hz)')
    plt.title('Filtro de síntesis')
    plt.figure(figsize=(10,1.5))
    plt.plot(n[Nha*(trama-1):Nha*(trama-1)+Nb]/fs,x[Nha*(trama-1):Nha*(trama-1)+Nb])
    plt.xlabel('t(s)')
    plt.title('Trama')
    plt.ylim(np.min(x),np.max(x))
    noverlap=Nb-Nha
    w=get_window(ventana,Nb)
    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.plot(x)
    plt.title('Forma de onda')
    plt.xlim(0,len(x))
    plt.xticks([],[])
    plt.axvspan(Nha*(trama-1),Nha*(trama-1)+Nb,color='orange',alpha=0.3)
    plt.subplot(312)
    plt.plot(P)
    plt.axvspan(trama-1,trama,color='orange',alpha=0.3)
    plt.title('Pitch')
    plt.ylabel('f(Hz)')
    plt.xlim(0,len(P))
    plt.xticks([],[])
    plt.subplot(313)
    plt.specgram(x, Fs=fs, NFFT=Nb, noverlap=int(noverlap), window=w); 
    plt.title('Espectrograma')
    plt.xlabel('t(s)')
    plt.ylabel('f(Hz)')
    plt.axvspan(Nha*(trama-1)/fs,(Nha*(trama-1)+Nb)/fs,color='orange',alpha=0.3)
    plt.xlim(0,len(x)/fs)

    
def analizavoz(x,T_trama,T_shift,ventana='hann',fs=8000):
    '''analizavoz(x,T_trama,T_shift,ventana='hann',fs=8000)
       Utilidad para analizar los resultados de una codificación de voz LPC '''
        
    P,G,A=codificadorlpc10(x,T_trama,T_shift,fs=fs,ventana=ventana)
    Nb=int(np.ceil(T_trama*fs)); 
    Nha=int(np.ceil(T_shift*fs));  
    P[P>0]=fs/P[P>0]
    interactive_plot=widgets.interactive(analizavoz_interactivo,A=widgets.fixed(A),x=widgets.fixed(x),trama=(1,len(G)),Nb=widgets.fixed(Nb),Nha=widgets.fixed(Nha),fs=widgets.fixed(fs),P=widgets.fixed(P),ventana=widgets.fixed(ventana))
    interactive_plot.children[0].continuous_update = False
    interactive_plot.children[0].value = 1
    display(interactive_plot)



def filtrosintonizador(f0,deltaf,fs=44100):    
    '''filtrosintonizador(f0,deltaf,fs=44100)
       Obtiene los coeficientes de un filtro IIR segundo orden sintonizado a la frecuencia f0 y de ancho de banda delta
        Argumentos de entrada:
            f0(float)-> frecuencia de sintonización en Hercios
            delta(float) ->  Ancho de banda del filtro en Hercios
            fs (float) -> frecuencia de muestreo
    
    Salida:
     a (np.ndarray)  -> Contiene los coeficientes recursivos  del filtro
     b (np.ndarray)  -> Contiene los coeficientes no recursivos  del filtro
    ''' 
    
    k=1/(1+np.tan(np.pi*deltaf/fs));
    
    b=[1-k,0,-(1-k)]
    a=[1,-2*k*np.cos(2*np.pi*(f0/fs)),(2*k-1)]
    
    return b, a


def filtra_armonicos(x,Tambloque,P,f_s):
    '''filtra_armonicos(x,Tambloque,P,f_s)
           utilidad interactiva para filtrar y escuchar los armónicos de la voz
    Argumentos de entrada:
            x (np.ndarray) -> Contiene las muestras de la señal de audio analizar
            Tambloque(int)-> escalar con la duración del bloque de reproducción en muestras
            P(np.ndarray) ->  Vector con los valores del Pitch en cada trama
            f_s (float) -> frecuencia de muestreo
    
    '''
    ver=widgets.__version__[0]
    global CHUNK,zo,fs
    fs=f_s
    CHUNK=int(Tambloque)
    zo=np.zeros(2)
    
    
    # CONTROLES:
    # Barra de tiempos
    barra=widgets.IntSlider(
    value=0,
    min=0,
    max=len(P)-1,
    step=1,
    description='',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=False,
    )
    
    # play-pause
    play = widgets.Play(
    value=0,
    min=0,
    max=len(P)-2,
    step=1,
    description="",
    disabled=False
    )
    if ver=='8':
        play.playing=False
        play.repeat=True
    else:
        play._playing=False
        play._repeat=True
    
    
    # Visor numérico
    numerico=widgets.FloatText(value=0.0, layout= widgets.Layout(width='65px'),disabled=True, readout_format='.4f')
    

    
    # Frecuencia resonancia filtro:
    Freq_Resonancia=widgets.FloatSlider(
    value=100.0,
    min=10,
    max=3000,
    step=0.1,
    description='Frecuencia Resonancia:',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    )
    
    Check_Freq=widgets.Checkbox(
    value=False,
    description='',
    disabled=False
    )

    
    Check_Freq=widgets.Checkbox(
    value=False,
    description='',
    disabled=False
    )

    Check_Espectro=widgets.Checkbox(
    value=False,
    description='Visualizar espectro trama',
    disabled=False
    )

    
    # Ancho de banda  filtro:
    Delta_w=widgets.FloatSlider(
    value=10.0,
    min=0.5,
    max=100,
    step=0.1,
    description='Ancho de banda:',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    )

    
    # Armónico
    armonico_slider=widgets.IntSlider(
    value=1,
    min=1,
    max=20,
    step=1,
    description='Armónico:',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    )
    
    Check_armonico=widgets.Checkbox(
    value=False,
    description='Sintonizar filtro a armónico',
    disabled=False
    )

    ## FIN DECLARACIÓN CONTROLES
    
    # FIGURA OSCILOSCOPIO
    xs = bq.LinearScale()
    ys = bq.LinearScale()
    
    ys.min=0
    ys.max=2.5

    
    ejex = np.arange(512)*fs/(2*512)
    y = np.zeros((2,CHUNK)) 

    line = bq.Lines(x=ejex, y=y, scales={'x': xs, 'y': ys}, colors=['red','blue'])
    xax = bq.Axis(scale=xs, label='x', grid_lines='solid')
    yax = bq.Axis(scale=ys, orientation='vertical', tick_format='0.2f', label='y', grid_lines='solid')

    fig = bq.Figure(marks=[line], axes=[xax, yax])
    
    # MOSAICO VISUALIZACIÓN
    #display(widgets.VBox([widgets.HBox([play, barra,numerico]),widgets.HBox([Gan_slider, Check_Gan]),widgets.HBox([estereo_slider, Check_estereo])]))
    controles=widgets.VBox([widgets.HBox([play, barra,numerico]),widgets.HBox([Freq_Resonancia, Check_Freq]),Delta_w,widgets.HBox([armonico_slider, Check_armonico])])
    display(widgets.VBox([Check_Espectro,fig,controles]))
    
    def on_value_changeplay(change):
        global reproduciendo,x,CHUNK,P        
        if reproduciendo==False:
            if play.value==0:
                barra.value=0        
            t1=threading.Thread(target=reproductor, args=(x,CHUNK,P,barra.value))    
            t1.start()

    def on_value_changebarra(change):
        global CHUNK,fs
        play.value=barra.value      
        numerico.value=(barra.value*CHUNK)/fs
        
    play.observe(on_value_changeplay, names='value')
    barra.observe(on_value_changebarra, names='value')
        
        


    def reproductor(x,CHUNK,P,trama=0):
        global reproduciendo,zo
        reproduciendo=True
        muestras = len(x)
        bloques=len(P) 
        
              

        p = pyaudio.PyAudio() # Objeto de audio
        stream = p.open(format=pyaudio.paInt16,
               channels=1,
               rate=fs,
               output=True)
        
        
        
        if ver=='8':
            reproducir = play.playing
        else:
            reproducir=play._playing 
                        
        
        
        while ((trama < bloques+1) ):
            
            if ver=='8':
                reproducir= play.playing
                repite = play.repeat
            else:
                reproducir=play._playing
                repite = play._repeat
            
            #if ((trama == bloques) and (play._repeat)):
            if ((trama == bloques) and (repite)):
                trama=0
        
            data=x[CHUNK*(trama):CHUNK*(trama+1)]
            # Procesado
            if Check_armonico.value:
                if P[trama]>0:
                    Freq_Resonancia.value=armonico_slider.value*fs/P[trama]
                    Freq_Resonancia.disabled=True
                
            else:
                Freq_Resonancia.disabled=False

            if Check_Freq.value:  # Calcula Filtro 
                fc=Freq_Resonancia.value
                coef_b,coef_a=filtrosintonizador(fc,Delta_w.value,fs)
                
            else:
                coef_a=np.array([1,0,0])
                coef_b=np.array([1,0,0])
            
            fd,H=freqz(coef_b,coef_a,fs=fs)    
            
            # Fin de procesado 
            if reproducir:
                salida,zo=lfilter(coef_b,coef_a,data,zi=zo)
                data=salida.astype(np.int16)

                # Reproducimos
                stream.write(bytes(data))   # Importante conversión a bytes    
                # Actualizamos controles
                barra.value=trama
                numerico.value=(trama*CHUNK)/fs
                trama=trama+1
            else:
                salida=lfilter(coef_b,coef_a,data)
                data=salida.astype(np.int16)
                
                trama=play.value
                if trama==0:
                    barra.value=0
            
            if Check_Espectro.value:
                fd,Xdata=freqz(salida/(25*np.max(np.abs(x))),1,fs=fs)
            
            else:
                Xdata = np.zeros(512)

            # Actualizamos gráficos
            if trama<bloques:
                line.x = fd
                linea1=np.zeros([1,512])
                linea1[0,:]=np.abs(H)
                linea2=np.zeros([1,512])
                linea2[0,:]=np.abs(Xdata)
                
                line.y = np.vstack([linea1,linea2])

                #line.y = np.abs(H)
                
            
            
    
        reproduciendo=False    
        
        if ver=='8':
            play.playing=False
        else:
            play._playing=False
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    

    t1=threading.Thread(target=reproductor, args=(x,CHUNK,P,0))    
    t1.start()

    
    
def espectrograma_tr(fs,B,NumBvisu=10,Nfft=128,Salto=1,x=np.array([None,None])):
    '''Representa gráficamente el espectrograma en tiempo real.

      Argumentos de entrada:
        fs (escalar): frecuencia de muestreo
        B (escalar): Tamaño del bloque de datos  
        NumBvisu (escalar): Número de bloques de datos representados en el eje temporal
        Nfft (escalar): Tamaño de la FFT y tamaño de las particiones de datos para el análisis frecuencial
        Salto (escalar): Número de bloques entre cada actualización de la representación gráfica 
        x: Señal a representar o NONE para analizar la señal captada por el micrófono'''
    
    global In,fss,CHUNK,xss,n,cont,numtramas,gNfft,Tramaanalisis,gNumB,Tramavisual,NumD,Ndibu
    
    CHUNK=B 
    Ndibu=NumBvisu
    numf=np.floor(Nfft/2)+1
    
    Tramaanalisis=np.zeros(int(3*B))
    In=False
   
    if x[0]==None:
        In=True
    else:
        numtramas=np.ceil(len(x)/CHUNK)
        xss=np.zeros(int(numtramas)*CHUNK)
        xss[0:len(x)]=x

  
    n=0
    cont=1
    gNfft=Nfft
    gNumbN=3
    gNumB=Salto
                 
    NumD=int(1+np.ceil(B*(gNumbN-2)/(Nfft/2)))
    Tramavisual=np.zeros((int(numf),int(NumBvisu*NumD)))
    
    fss=fs

    # Botones y controles
    botonDet=widgets.Button(
            description='Detener',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    botonEmp=widgets.Button(
            description='Empezar',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    
    CheckA=widgets.Checkbox(
    value=False,
    description='dB',
    disabled=False,
    indent=False
    )
    
    Checkf=widgets.Checkbox(
    value=False,
    description='Freq. log',
    disabled=False,
    indent=False
    )
    
    anchobanda=widgets.FloatRangeSlider(
    style = {'description_width': 'auto'},
    value=[0, fs/2],
    min=0,
    max=fs/2,
    step=fs/Nfft,
    description='Banda Visualizada:',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.1f',
    layout=widgets.Layout(width='95%', height='300px'),
    )

            # FIGURA ESPECTROGRAMA   
    
    layoutfig=widgets.Layout(width="700px", height="300px")
    
    x_sc, y_sc, col_sc = bq.LinearScale(), bq.LinearScale(), bq.ColorScale() 
    y_sc.max=fs/2
    
    espectro = bq.HeatMap(y=np.arange(numf)*fs/(2*numf),color=Tramavisual,  scales={'x': x_sc, 'y': y_sc, 'color': col_sc}, shading='gouraud')
    ax_x = bq.Axis(scale=x_sc,visible=False) 
    ax_y = bq.Axis(scale=y_sc, orientation='vertical')
    col_sc.scheme='OrRd'
    fig2 = bq.Figure(marks=[espectro], padding_y=0.0,axes=[ax_x, ax_y],layout=layoutfig)  
   
        # MOSAICO VISUALIZACIÓN
    display(widgets.VBox([widgets.HBox([fig2,anchobanda]),widgets.HBox([botonEmp, botonDet, CheckA, Checkf])]))
    
    # Instancia a clase pyaudio
    p = pyaudio.PyAudio()
    
    def callback(in_data, frame_count, time_info, status):
        global In,fss,CHUNK,xss,n,numtramas,cont,Tramaanalisis,gNumB
        flag= pyaudio.paContinue
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        if In==False:
            if n==numtramas*CHUNK:
                n=0
            in_data=xss[n:n+CHUNK]
            n=n+CHUNK
        else:
            in_data=audio_data
        
        Tramaanalisis=np.concatenate((Tramaanalisis[CHUNK:],in_data))    
        if cont==gNumB:
            cont=1
            dibujaespectro(Tramaanalisis)
        else:
            cont=cont+1
                
        return (bytes((not(In))*in_data.astype(np.int16)),flag)
        
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fss,
                input=True,
                output=True,
                start=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)    
    
    def on_button_clickedDet(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True

    def on_button_clickedEmp(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True       
        

    def dibujaespectro(tramaanalisis):
        global fss,gNfft,Tramavisual,NumD,Ndibu
        f, t, Zxx = signal.stft(tramaanalisis, fss, nperseg=gNfft)
        Tramavisual[:,:NumD*(Ndibu-1)]=Tramavisual[:,NumD:]
        Tramavisual[:,NumD*(Ndibu-1):]=np.abs(Zxx[:,NumD-1:-NumD+1])
        fmin=np.floor(numf*anchobanda.value[0]/(fs/2))
        fmax=np.floor(numf*anchobanda.value[1]/(fs/2))
        
        if CheckA.value==True:
            espectro.color=10*np.log10(Tramavisual[int(fmin):int(fmax)]+0.00001)
        else:
            espectro.color=1*Tramavisual[int(fmin):int(fmax)]
            
        if Checkf.value==True:
            y_sc.max=np.log2(anchobanda.value[1])
            y_sc.min =np.log2(0.001+anchobanda.value[0])
            espectro.y=np.log2(0.001+anchobanda.value[0]+np.arange(fmax-fmin)*anchobanda.value[1]/(fmax-fmin))
        else:
            y_sc.max=anchobanda.value[1]
            y_sc.min=anchobanda.value[0]
            espectro.y=anchobanda.value[0]+np.arange(fmax-fmin)*anchobanda.value[1]/(fmax-fmin)
        
    botonDet.on_click(on_button_clickedDet)
    botonEmp.on_click(on_button_clickedEmp)



    
def dibujaespec(X,fs,escala='dB',titulo='Espectrograma'):
    """ Representa gráficamente el espectrograma entiquecido con información en los ejes y título.

      Argumentos de entrada:
        X (ndarray [f,t]): Matriz de dos dimensiones con los valores del espectrograma (en módulo)
        fs (escalar): frecuencia de muestreo 
        escala (string): cadena para indicar si se desea una representación en unidades logarítmicas (por defecto) o lineales (cualquier valor diferente a 'dB')
        titulo (string): Cadena con la etiqueta que desea añadirse al encabezado de la figura """
    
    if escala=='dB':
        P= librosa.amplitude_to_db(X, ref=np.max)
    else:
        P=X
    fig, ax = plt.subplots()
    img = librosa.display.specshow( P , sr=fs, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=titulo)
    if escala=='dB':
        fig.colorbar(img, ax=ax, format="%+2.f dB");
    else:
        fig.colorbar(img, ax=ax, format="%+2.f");
        

def Compara_matrices_binarias(C_ref, C_est, tol_freq=0, tol_time=0):
    """ Compara matrices binarias considerando cierta tolerancia
    
    Argumentos de entrada:
        C_ref (np.ndarray): Matriz binaria usada como referencia
        C_est (np.ndarray): Matriz binaria usada como estimación
        tol_freq (int): Tolerancia en dirección vertical (frequencia) (Por defecto = 0)
        tol_time (int): Tolerancia en dirección horizontal (tiempo) (Por defecto = 0)

    Salida:
        TP (int): Coincidencias
        FN (int): Falsos negativos
        FP (int): Falsos positivos
        C_AND (np.ndarray): Máscara Booleana  de la intersección lógica de C_ref y C_est (con tolerancia)
    """
    assert C_ref.shape == C_est.shape, "Las dimensiones tienen que coincidir"
    N = np.sum(C_ref)
    M = np.sum(C_est)
    # Expandimos C_est con un filtro de máximo 2D usando una región definida por los parámetros de tolerancia
    C_est_max = ndimage.maximum_filter(C_est, size=(2*tol_freq+1, 2*tol_time+1),
                                       mode='constant')
    C_AND = np.logical_and(C_est_max, C_ref)
    TP = np.sum(C_AND)
    FN = N - TP
    FP = M - TP
    return TP, FN, FP, C_AND


def calcula_similitud(C_D, C_Q, tol_f=1, tol_t=1):
    """Calcula la similitud entre dos mapas de constelaciones

    Argumentos de entrada:
        C_D (np.ndarray): Matriz binaria con el mapa de constelación del audio de referencia (perteneciente a una base de datos)
        C_Q (np.ndarray): Matriz binaria con el mapa de constelación del estracto de audio a identificar 
        tol_f (int): Tolerance in frequency direction (vertical) (Default value = 1)
        tol_t (int): Tolerance in time direction (horizontal) (Default value = 1)

    Salida:
        Delta (np.ndarray): Función de similitud
        shift_max (int): Posición de desplazamiento óptimo que maximiza la función de similitud
    """
    L = C_D.shape[1]
    N = C_Q.shape[1]
    M = L - N
    assert M >= 0, "El audio a identificar debe ser más corto que el de referencia"
    Delta = np.zeros(L)
    for m in range(M + 1):
        C_D_crop = C_D[:, m:m+N]
        TP, FN, FP, C_AND = Compara_matrices_binarias(C_D_crop, C_Q,
                                                      tol_freq=tol_f, tol_time=tol_t)
        Delta[m] = TP
    shift_max = np.argmax(Delta)
    return Delta, shift_max
 
def calcula_similitud_interactiva(mapaCr, mapaCq, n=0, tol_f=0, tol_t=0):
    
    mapaCq= ndimage.maximum_filter(mapaCq, size=(2*tol_f+1, 2*tol_t+1),
                                       mode='constant')
    similitud, desplazamiento=calcula_similitud(mapaCr, mapaCq, tol_f=tol_f, tol_t=tol_t)
    
    fig, ax = plt.subplots(1, 1, figsize=(7,3))

    ax.set_ylabel('frecuencia (bin)');
    ax.set_xlabel('tiempo(muestras)');
    k1, n1 = np.argwhere(mapaCq == 1).T
    k2, n2 = np.argwhere(mapaCr == 1).T
    ax.scatter(n2, k2, color='r', s=3, marker='o');
    ax.scatter(n+n1, k1, color='b', s=1, marker='o');
    ax.set_title('Similitud: %d' % int(similitud[n]));    

def argprin(v):
    """Función argumento principal

   
    Argumentos de entrada:
        v (float o np.ndarray): Valor (o vector de valores) con las fases 

    Salida:
        w (float o np.ndarray): Argumento principal de v
    """
    w = (np.mod(v/(2*np.pi) + 0.5, 1) - 0.5)*2*np.pi
    return w

def encuentra_fi_espectrograma(X , fs, N, H, umbral=30, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.01, wait=2):
    """Encuentra los picos del espectrograma por encima de un umbral en relación al valor máximo. Se basa en la función librosa.util.peak_pick.
     Sobre esas frecuecias se calcula la frecuencia instantánea (proceso de refinamiento)
     Argumentos de entrada:
         X (ndarray[f,n]) con los valores de la STFT
         fs (escalar): frecuencia de muestreo
         N (int): Tamaño de la ventana en muestras
         H (int): Tamaño de salto
         umbral (escalar): valores en dB respecto del máximo para considerar los picos válidos
         
         
         El resto de argumentos sirven para configurar la función librosa.util.peak_pick.
         pre_max(escalar): número de puntos anteriores a considerar en la ventana de búsqueda
         post_max(escalar): número de puntos posteriores a considerar en la ventana de búsqueda
         pre_avg(escalar): número de puntos anteriores a considerar en la ventana de promedio
         post_avg(escalar): número de puntos posteriores a considerar en la ventana de promedio
         delta(escalar): umbral sobre el valor medio
         wait(escalar): número de puntos mínimo entre máximos detectados
         

    Salida:
         fi(ndarray[f,n]): Matriz con valor  de la frecuencia instantánea asociada a cada bin donde hay un máximo

    """ 
    nf,nt=np.shape(X)
    picos=np.zeros([nf,nt])
    fi=np.zeros([nf,nt])
    Xm=np.abs(X)
    max=20*np.log10(np.max(np.max(Xm)))
    lmin=10**((max-umbral)/20)
    
    phi_1 = np.angle(X[:, 0:-1])
    phi_2 = np.angle(X[:, 1:]) 
    index_k = np.arange(0, nf).reshape(-1, 1)
    kappa =  argprin(phi_2 - phi_1 - 2*np.pi*index_k * H / N)
    F_coef_IF = (2*np.pi*index_k*H/N + kappa) * fs /(2*np.pi*H) 
    
    
    # Extendemos F_coef_IF copiando la última columna para obtener las mismas dimensiones que X
    F_coef_IF = np.hstack((F_coef_IF,np.copy(F_coef_IF[:, -1]).reshape(-1, 1)))
    
    for t in range(nt):
        peaks = librosa.util.peak_pick(Xm[:,t], pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
        if len(peaks) !=0:
            peaku=np.where(Xm[peaks,t]>lmin)
            if len(peaku) != 0:
                
                picos[peaks[peaku],t]=Xm[peaks[peaku],t]
                fi[peaks[peaku],t]=F_coef_IF[peaks[peaku],t]
    

    return picos,fi

def estimaf0(fi):
    """Devuelve un único valor de la frecuencia fundamental por cada columna, a partir de una matriz que contiene en cada columna diferentes frecuencias cadidatas a Pitch. Cada columna representa un instante temporal. En caso de no tener frecuencias candidatas en alguna columna, se devuelve un cero. La frecuencia candidata devuelta es la frecuencia menor (mayor que cero)
    
    Argumentos de entrada:
         fi (ndarray[f,n]) matriz con las frecuencias candidatas a pitch para cada instante temporal en cada columna 
                  

    Salida:
         fo(ndarray[1,n]): Vector con la frecuencia instantánea mínima (mayor que cero) de cada columna

    """ 
    nk,nn=np.shape(fi)
    f0=np.zeros(nn)
    for n in range(nn):
        if np.sum(fi[:,n])>0:
            k = np.argwhere(fi[:,n] > 0)
            f0[n]=np.min(fi[k,n],0)
    return f0

def segmentaf0(fo,M=3,D=3,umbral=10):
    """Segementa e interpola la trayectoria del pitch descrita en f0. Considera que una región de sonido sordo o sonoro debe tener M valores consecutivos de la misma naturaleza. Interpola los valores de las regiones sonoras y corrige los valores de pitch atípicos
    
    Argumentos de entrada:
         fo (ndarray[1,n]) vector con la trayectoria del f0
         M (escalar) mínimo número de valores consecutivos para considerar los segmentos
         D (escalar) mitad del ancho del filtro de promedio para deterctar pitch anómalos
         umbral (escalar) valor diferencial que ha de superar un valor de pitch sobre el promedio en torno a D valores, para considerar el pitch como anómalo. También es el valor que se usa para determinar la variación máxima permitida del pitch en tramos sonoros

    Salida:
         fo(ndarray[1,n]): Vector con la trayectoria de f0 segmentada y corregida

    """ 
    # Definimos el filtro de promediado
    if D>M:
        D=M
    h=np.ones(2*D+1)/(2*D)
    h[D+1]=0
    
    # Realizamos una doble diferenciacio para detectar grandes variaciones en el pitch y las hacemos cero (sonidos sordos)
    f0=1*fo
    ruido=np.argwhere(np.abs(np.diff(f0))>umbral)
    f0[ruido]=0
    ruido=np.argwhere(np.abs(np.diff(f0))>umbral)
    f0[ruido]=0
    
    Nbloques=len(f0)
    sonoras=np.zeros(Nbloques+1)
    
    # Eliminamos los tramos sordos muy cortos interpolando
    sonoras[0:Nbloques]=np.int8(f0>0)
    sonoras[-1]=np.int8(not(sonoras[-2]))

    cambio=np.diff(sonoras)
    bloquecambio=np.argwhere(cambio != 0)

    inicio=0
    for n in range(len(bloquecambio)):
        fin=1+bloquecambio[n]
        if fin-inicio<M:
            if not(sonoras[fin-1]):
                f0[int(inicio-1):int(fin+1)]=np.linspace(f0[int(inicio-1)],f0[int(fin)],len(f0[int(inicio-1):int(fin+1)]))
            
        inicio=fin   
    
    # Eliminamos los tramos sonoros muy cortos 
    sonoras[0:Nbloques]=np.int8(f0>0)
    sonoras[-1]=np.int8(not(sonoras[-2]))

    cambio=np.diff(sonoras)
    bloquecambio=np.argwhere(cambio != 0)

    inicio=0
    for n in range(len(bloquecambio)):
        fin=1+bloquecambio[n]
        if fin-inicio<M:  
            if (sonoras[fin-1]):
                f0[int(inicio):int(fin)]=0
        inicio=fin   

    # Eliminamos anomalías
    sonoras[0:Nbloques]=np.int8(f0>0)
    sonoras[-1]=np.int8(not(sonoras[-2]))

    cambio=np.diff(sonoras)
    bloquecambio=np.argwhere(cambio != 0)

    inicio=0

    for n in range(len(bloquecambio)):
        fin=1+bloquecambio[n]
        if (sonoras[fin-1]):
            Bloquesonoro=f0[int(inicio):int(fin)]                    
            Bloquesonoro_e=np.concatenate([Bloquesonoro[D:0:-1],Bloquesonoro,Bloquesonoro[-2:-2-D:-1]])
            Bloquesonoro_e=signal.convolve(Bloquesonoro_e,h)
            Bloquesonoro_e=Bloquesonoro_e[2*D:-2*D]
            anomalias=np.argwhere(np.abs(Bloquesonoro_e-Bloquesonoro)>umbral)
            Bloquesonoro[anomalias]=Bloquesonoro_e[anomalias]
            f0[int(inicio):int(fin)]=Bloquesonoro
        inicio=fin   
    return f0 

def sintetizaf0(f0,A,fs,H):
    ''' Sintetiza con un oscilador la evolución de una trayectoria de Pitch contenida en f0 con amplitudes asociadas en A.
    
    Argumentos de entrada:
         f0 (ndarray[1,n]) vector con la trayectoria del f0
         A ndarray[1,n]) vector con las amplitudes a usar en la sintetización 
         fs (escalar) frecuencia de muestreo
         H (escalar) número de muestras de salto

    Salida:
         y (ndarray[1,n·H]): Sonido sintetizado
    '''
    
    y=np.zeros(len(f0)*H)
    A0=0
    f_n=0
    t=np.arange(H)
    fi0=0
    for n in range(len(f0)):
        Amp=np.linspace(A0,A[n],H)
        if n>0:
            fi0=2*np.pi*(t[0])*(f_n-f0[n])/(fs)+fi0
        y[0+H*(n):H*(n+1)]=Amp*np.cos(2*np.pi*(f0[n]/fs)*t+fi0)
        A0=A[n]
        f_n=f0[n]
        t=t+H
    return y

def trayectoriaPitch_stft(x,fs,NFFT,H,fmax,umbral=30, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.01, wait=2):
    ''' Calcula la trayectoria del pitch, la trayectoria con las frecuencias de las notas musicales más cercanas y sus amplitudes.
    
    Argumentos de entrada:
         x ndarray [1,n] con la forma de onda de la señal de audio  
         fs (escalar) frecuencia de muestreo
         NFFT (escalar) número de muestras del bloque (y de la FFT para el análisis frecuencial) 
         H (escalar) número de muestras de salto
         fmax (escalar) frecuencia máxima a considerar en el análisis
         umbral (escalar): valores en dB respecto del máximo para considerar los picos válidos
         
         
         El resto de argumentos sirven para configurar la función librosa.util.peak_pick.
         pre_max(escalar): número de puntos anteriores a considerar en la ventana de búsqueda
         post_max(escalar): número de puntos posteriores a considerar en la ventana de búsqueda
         pre_avg(escalar): número de puntos anteriores a considerar en la ventana de promedio
         post_avg(escalar): número de puntos posteriores a considerar en la ventana de promedio
         delta(escalar): umbral sobre el valor medio
         wait(escalar): número de puntos mínimo entre máximos detectados


    Salida:
         f0 (ndarray): Trayectoria del Pitch en los instantes múltiplos del tamaño de salto
         f0nota (ndarray): Trayectoria de las frecuencias de las notas musicales en los instantes múltiplos del tamaño de salto
         a: amplitudes asociadas a las frecuencias de la trayectoria
    '''
    
    noverlap=NFFT-H
    f,t,Zxx=signal.stft(x,fs=fs,nperseg=NFFT,noverlap=noverlap);
    kmax=int(np.ceil(fmax*NFFT/fs))
    picos,fi=encuentra_fi_espectrograma(Zxx[0:kmax,:], fs, NFFT, H, umbral=umbral, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
    nk,nn=np.shape(fi)
    fpitch=np.zeros(nn)
    for n in range(nn):
        if np.sum(fi[:,n])>0:
            k = np.argwhere(fi[:,n] > 0)
            fpitch[n]=np.min(fi[k,n],0)
    f0=segmentaf0(fpitch)
    n=np.argwhere(f0>0) # Bloques con Pitch válido
    a=np.zeros(len(f0))
    indices_k=np.int16(np.rint(f0[n]*NFFT/fs))
    a[n]=np.abs(Zxx[indices_k,n]) # Amplitud asociada a cada frecuencia fundamental
    pfref=np.zeros(len(f0))
    pfref[n]=np.rint(12*np.log2(f0[n]/440)+69)
    f0nota=np.zeros(len(f0))
    f0nota[n]=440*2**((pfref[n]-69)/12) # Calculamos las frecuencias nominales de las notas musicales más cercanas al pitch

    return f0,f0nota,a

def errorpredlin(trama,P=10):
    ''' devuelve la señal de error de prediccion de un filtro de predicción lineal de orden P 
    Argumentos de entrada:
        trama (ndarray[n,1]): con las señal sobre la cual calcular el filtro LPC y el error de predicción
        P (escalar): Orden del filtro de predicción a considerar
        
    Salida:
        error (ndarray[n,1]): vector con la señal de error de predicción
    '''
    a,Pe= lpc(trama,P)
    error=signal.lfilter(a,1,trama)
    
    return error

def c_clip(trama,c):
    ''' devuelve la señal de error recortada de forma que todos los valores que estén por debajo de un umbral determinado por c se saturan a cero y al resto se les resta dicho umbral 
    Argumentos de entrada:
        trama (ndarray[n,1]): con las señal sobre la cual calcular el filtro LPC y el error de predicción
        c (escalar): Factor respecto del máximo a considerar como umbral
        
    Salida:
        cclip (ndarray[n,1]): vector con la señal recortada
    '''
    umbralp=np.max(trama)
    umbraln=np.min(trama)
    xclip=np.clip(trama,c*umbraln,c*umbralp)
    
    cclip=trama-xclip

    
    return cclip

def trayectoriaf0_xcorr(x,fs,fmax=2093,L=2048,H=512,  C=0, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.01, wait=2):
    ''' Calcula la trayectoria del pitch, usando la correlación y localizando el segundo pico máximo.
    
    Argumentos de entrada:
         x ndarray [1,n] con la forma de onda de la señal de audio  
         fs (escalar) frecuencia de muestreo
         fmax (escalar) frecuencia máxima a considerar para el Pitch
         L (escalar) número de muestras del bloque
         H (escalar) número de muestras de salto
         C (escalar) factor de saturación para aplicar saturación central        
         
         El resto de argumentos sirven para configurar la función librosa.util.peak_pick.
         pre_max(escalar): número de puntos anteriores a considerar en la ventana de búsqueda
         post_max(escalar): número de puntos posteriores a considerar en la ventana de búsqueda
         pre_avg(escalar): número de puntos anteriores a considerar en la ventana de promedio
         post_avg(escalar): número de puntos posteriores a considerar en la ventana de promedio
         delta(escalar): umbral sobre el valor medio
         wait(escalar): número de puntos mínimo entre máximos detectados


    Salida:
         f0 (ndarray): Trayectoria del Pitch en los instantes múltiplos del tamaño de salto
        
    '''
    
    Numblock=int(1+np.floor((len(x)-L)/H))
    f0=np.zeros(Numblock)
    f0[0:]=np.nan
    for ini in range(Numblock):
        Trama1=c_clip(x[ini*H:ini*H+L],C)
        Rx1=signal.correlate(Trama1,Trama1)
        Rxp=Rx1[L-1:]
        peaks = librosa.util.peak_pick(Rxp, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta= delta, wait=wait)
        if len(peaks)>1:
            picos_ordenados=np.sort(Rxp[peaks])
            M=np.argwhere(Rxp==picos_ordenados[-2])
            if fs/M[0]<fmax:
                f0[ini]=fs/M[0]
    return f0

def trayectoriaf0_xcorre(x,fs,fmax, P, L=2048,H=512,C=0,Umbral=0):
    ''' Calcula la trayectoria del pitch, usando la correlación de la señal de error de predicción y localizando el segundo máximo.
    
    Argumentos de entrada:
         x ndarray [1,n] con la forma de onda de la señal de audio  
         fs (escalar) frecuencia de muestreo
         fmax (escalar) frecuencia máxima a considerar para las frecuencias candidatas a Pitch
         P (escalar) orde del filtro de predicción
         L (escalar) número de muestras del bloque
         H (escalar) número de muestras de salto
         C (escalar) factor de saturación para aplicar saturación central
         Umbral (escalar): Factor umbral entre la amplitud del pico y la potencia media de la trama para la decisión entre sonora y sorda
                  
         
        
    Salida:
         f0 (ndarray): Trayectoria del Pitch en los instantes múltiplos del tamaño de salto
        
    '''
    
    Numblock=int(1+np.floor((len(x)-L)/H))
    f0=np.zeros(Numblock)
    f0[0:]=np.nan
    ind=int(np.floor(fs/fmax))
    for ini in range(Numblock):
        Trama1=c_clip(x[ini*H:ini*H+L],C)
        error1=errorpredlin(Trama1,P)
        Rxe=signal.correlate(error1[P:],error1[P:]) # Descartamos las P primeras muestras para evitar efectos de transitorios
        Rxep=Rxe[L-P-1:]
        M=np.argwhere(Rxep==np.max(Rxep[ind:]))
        if Rxep[M]/Rxep[0]>Umbral:
            f0[ini]=fs/M
    return f0

def xcorr_norm(xp,lmin,lmax,NBlock):
    ''' Calcula la autocorrelación normalizada de un bloque en un rango determinado por lmin y lmax.
    
    Argumentos de entrada:
         x ndarray [1,n] con la forma de onda de la señal de audio  
         fs (escalar) frecuencia de muestreo
         fmax (escalar) frecuencia máxima a considerar para las frecuencias candidatas a Pitch
         fmin (escalar) frecuencia mínima a considerar para las frecuencias candidatas a Pitch
         L (escalar) número de muestras del bloque
         H (escalar) número de muestras de salto
         b0_th (escalar): Umbral en torno a la unidad para considerar válido el coeficiente del filtro de predicción a largo plazo
                  
         
        
    Salida:
         rxx_norm (ndarray[1,lmax-lmin]): autocorrelación normalizada
         rxx (ndarray[1,lmax-lmin]): autocorrelación 
         rxx0 (ndarray[1,lmax-lmin]): autocorrelación en el origen (energía) 
        
    '''    
    xp=xp.astype(float)
    bloque=xp[lmax:NBlock+lmax]
    rango=np.arange(lmin,lmax+1)
    rxx=np.zeros(len(rango))
    rxx0=np.zeros(len(rango))
    rxx_norm=np.zeros(len(rango))
    
    for l in range(len(rango)):
        rxx0[l]=np.sum(xp[lmax-rango[l]:NBlock+lmax-rango[l]]**2)
        rxx[l]=np.sum(xp[lmax-rango[l]:NBlock+lmax-rango[l]]*bloque)
        
    rxx_norm=(rxx**2)/rxx0
    
    return rxx_norm, rxx, rxx0
       
    
def trayectoriaf0_LTP(x,fs,fmax=2093,fmin=65,L=2048,H=512,b0_th=0.2):
    ''' Calcula la trayectoria del pitch, usando la correlación a largo plazo.
    
    Argumentos de entrada:
         x ndarray [1,n] con la forma de onda de la señal de audio  
         fs (escalar) frecuencia de muestreo
         fmax (escalar) frecuencia máxima a considerar para las frecuencias candidatas a Pitch
         fmin (escalar) frecuencia mínima a considerar para las frecuencias candidatas a Pitch
         L (escalar) número de muestras del bloque
         H (escalar) número de muestras de salto
         b0_th (escalar): Umbral en torno a la unidad para considerar válido el coeficiente del filtro de predicción a largo plazo
                  
         
        
    Salida:
         f0 (ndarray[1,n]): Trayectoria del Pitch
        
    '''
    lmin=int(np.floor(fs/fmax))
    lmax=int(np.floor(fs/fmin))
    
    lags=np.arange(lmin,lmax+1)
    Nlag=len(lags)
    
    
    blocks=int(np.floor((len(x)-lmax-L)/H))
    
    f0=np.zeros(blocks)
    f0[0:]=np.nan
    for b in range(blocks):
        bloque=x[b*H:b*H+L+lmax]
                
        rxx_norm,rxx,rxx0=xcorr_norm(bloque,lmin,lmax,L)
       
        B0=rxx/rxx0

        peaks = librosa.util.peak_pick(rxx_norm,pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta= 2, wait=2)
       
        indices=np.argwhere(rxx[peaks]>0)
        indices2=peaks[indices]
       
        indices=np.argwhere(np.abs(B0[indices2]-1)<b0_th)
        
        indices2=indices2[indices[:,0]]
        M=lags[indices2]
        
        if np.any(M):
        
            f0[b]=fs/M[0]
    return f0

def trayectoriaPitch_cepstrum(x,fs,NFFT,H,fmin=50,fmax=300):
    ''' Calcula la trayectoria del pitch, la trayectoria con las frecuencias de las notas musicales más cercanas y sus amplitudes.
    
    Argumentos de entrada:
         x ndarray [1,n] con la forma de onda de la señal de audio  
         fs (escalar) frecuencia de muestreo
         NFFT (escalar) número de muestras del bloque (y de la FFT para el análisis frecuencial) 
         H (escalar) número de muestras de salto
         fmin (escalar) frecuencia mínima a considerar en el análisis
         fmax (escalar) frecuencia máxima a considerar en el análisis
         
         

    Salida:
         f0 (ndarray): Trayectoria del Pitch en los instantes múltiplos del tamaño de salto
         f0nota (ndarray): Trayectoria de las frecuencias de las notas musicales en los instantes múltiplos del tamaño de salto
         a: amplitudes asociadas a las frecuencias de la trayectoria
    '''
    
    lmin=int(np.floor(fs/fmax))
    lmax=int(np.min([NFFT/2,np.floor(fs/fmin)]))
    noverlap=NFFT-H
    
    f,t,Zxx=signal.stft(x,fs=fs,nperseg=NFFT,noverlap=noverlap,return_onesided=False);
    nk,nn=np.shape(Zxx)
    fpitch=np.zeros(nn)
    for n in range(nn):
        Xlog=np.log(np.abs(Zxx[:,n]))
        c=np.abs(np.real(np.fft.ifft(Xlog)))
        nmax=np.argwhere(c[0:lmax]==np.max(c[lmin:lmax]))
        fpitch[n]=fs/nmax

    return fpitch

def modificaPitch(xin,fs,B,H,f0,f02,fmin):
    """Devuelve la señal con un pitch original definido en f0, aplicándoles la transformación de pitch para que se corresponda con la trayectoria definida en f02.
     Argumentos de entrada:
         xin (ndarray[n,1]): Vector con la forma de onda a transformar
         fs (escalar): frecuencia de muestreo
         B (escalar): Tamaño de bloque
         H (escalar): Tamaño de salto
         f0 (ndarray): vector con el valor de la trayectoria original del pitch. Usa valores NaN para definir bloques no sonoros
         f02 (ndarray): vector con el valor de la trayectoria deseada del pitch. 
         fmin (escalar): vector con la frecuencia mínima a considerar (se usa en las tramas no sonoras)
         

    Salida:
         A_corrected (ndarray[n,1]): Vector con la forma de onda de la señal transformada

    """     
    pitchvalido=np.argwhere(np.isfinite(f02))
    f02min=np.min(f02[pitchvalido])
    if f02min<fs/B:  # Comprobamos que el tamaño del bloque es apropiado para el rango de valores de pitch
        print('Para los valores de pitch introducidos, el tamaño de bloque debe ser mayor que', fs/f02min)
        sys.exit()
    
    fmin=np.max([fmin,fs/B])  # Reajustamos la frecuencia mínima si su periodo es mayor que el número de muestras del bloque
    pitchvalido=np.argwhere(np.isfinite(f0))
    alfamax=np.max(f02[pitchvalido]/f0[pitchvalido])
    
    
    x=np.zeros(int((len(f0))*H+(1+alfamax)*B))
    x[0:len(xin)]=xin # Añadimos ceros al final para prevenir el desbordamiento en el caso de valores de f02/f0 altos en el último bloque
    
    #A_corrected=np.zeros(len(x))
    A_corrected=1*x  # Inicializamos la señal transformada con la señal original (se puede inicializar con ceros)
    alfa=np.zeros(len(f0))
    max_acorr_shift = 0;
    max_acorr_amp = 0;  

    for bloque in range(len(f0)):  # Recorremos todos los bloques
        A=x[bloque*H:bloque*H+B]
        tbloque=np.arange(bloque*H,bloque*H+B)/fs
        tbloque_i=1*tbloque
        A_i=1*A
        if np.isnan(f0[bloque]):  # En caso de bloques no sonoros
            alfa[bloque]=1  
            Nperiod = int(np.ceil(fs/fmin))
            Ainterp=1*A
            
        else:   # En caso de bloques sonoros
                   
            alfa[bloque]= f02[bloque]/f0[bloque] 
            Nperiod1 = int(np.ceil(1/(f0[bloque]/fs)))
            Nperiod =int(np.ceil(1/(f02[bloque]/fs)))
            tbloque2=1*tbloque
            tbloque_i=np.mean(tbloque) + (tbloque-np.mean(tbloque))*alfa[bloque]
            # Interpolamos
                        
            if alfa[bloque]>1:  # En caso de que tengamos que interpolar con más datos que el tamaño del bloque, rellenamos con periodos de la señal
                
                tbloque2=np.arange(np.min(tbloque_i),np.max(tbloque_i)+1/fs,1/fs)
                A_i=np.zeros(len(tbloque2))
                exceso=int(np.ceil((len(tbloque2)-B)/2))
                
                # Rellenamos periodos inferiores
                periodo0 = A[0:Nperiod1]
                if B<Nperiod1:  # Comprobamos que el tamaño del bloque es apropiado para el rango de valores de pitch
                    print('Tamaño de bloque pequeño para los valores de pitch introducidos')
                    sys.exit()
                veces=int(np.ceil(exceso/Nperiod1))
                resto=Nperiod1-exceso%Nperiod1
                if resto==Nperiod1:
                    resto=0
                a_exceso=np.tile(periodo0,veces)
                A_i[0:exceso]=a_exceso[resto:]
                
                # parte central
                A_i[exceso:exceso+B]=A
                
                # Rellenamos periodos superiores
                periodof=A[-Nperiod1:]
                a_exceso=np.tile(periodof,veces)
                A_i[exceso+B:]=a_exceso[0:len(A_i[exceso+B:])]
            
            
            f=interpolate.interp1d(tbloque2,A_i) # Calculamos la función de interpolación
            Ainterp=f(tbloque_i)   # interpolamos
            
            
        
        
        if bloque == 0:   # El primer bloque lo colocamos tal cual
            A_corrected[bloque*H:bloque*H+B] = Ainterp
        
        else:   # Lo siguientes bloques los enlazamos manteniendo la continuidad
        
        # Extraemos un periodo de la nueva forma de onda
            
            Achunk = Ainterp[0:Nperiod]
            factor = np.sum(np.abs(Achunk))**2;
        
            if not((alfa[bloque-1]==1 or alfa[bloque]==1)):
                # Inicializamos variables para el cálculo de la correlación
                max_acorr_amp = 0;
                max_acorr_shift = 0;
                minrango=-int(np.rint(Nperiod/2))
                maxrango=-int(np.rint(Nperiod/2))+Nperiod
                for Nshift in range(minrango,maxrango):

                # Calculamos el desplazamiento correspondiente al máximo de la  autocorrelación
                    acorr = 1 - np.sum(np.abs(Achunk  -  A_corrected[bloque*H+ Nshift :bloque*H + Nshift +Nperiod]))/factor

                    if acorr > max_acorr_amp:
                        max_acorr_amp = acorr
                        max_acorr_shift = Nshift
            

            n_ini = np.argmin(np.abs(Achunk -  A_corrected[bloque*H+ max_acorr_shift :bloque*H+ max_acorr_shift +Nperiod])) # indice para 'enganchar' los dos bloques
            A_corrected[bloque*H + max_acorr_shift  + n_ini : bloque*H + max_acorr_shift + B] = Ainterp[n_ini:]
        
    return A_corrected    


def ftobark(fhz):
    """ Conversión Hercios a Bark """
    bark=13*np.arctan(0.76*fhz/1000)+3.5*np.arctan((fhz/(7.5*1000))**2)
    return bark

def barktof(bark):    
    """ ConversiónBark Hercios"""
    f=1000*10**((bark-8.7)/14.2)
    return f

def model_psicoacustico_1(audio,b=16,fs=44100):

    """ Análisis psicoacústico de una trama de señal según el modelo de baja complejidad descrito en el MPEG-1.
    Argumentos de entrada:
    audio (ndarray): Trama de enteros con las 512 muestras a analizar
    b (escalar): Número de bits con los que se cuantificó la trama
    fs (escalar): frecuencia de muestreo
    
    Argumentos de salida:
    p: Densidadad espectral de la trama analizada
    th: Umbral de audiobilidad en reposo
    umbralporbanda: Umbral de enmascaramiento por banda
    SMR: Relación señal a enmascaramiento por banda"""
    


    N = 512 # Tamaño de la trama y de la FFT

    f_ = np.arange(N/2)  # inicializamos los índices frecuenciales de un semiperiodo frecuencial (frecuencias positivas)
    f_hz = np.arange(1, (N/2+1))*(fs/N) # Calculamos las frecuencias analógicas correspondientes ( resolución frecuencial de 86.13Hz )
    freq_bark = ftobark(f_hz) # conversión del rango frecuencial a escala Bark

    th = 3.64*((f_hz/1000)**-0.8)-6.5*np.exp(-0.6*((f_hz/1000-3.3)**2))+(10**-3)*((f_hz/1000)**4) # Fórmula del umbral de audioción
    th[187:256] = 69.13 

    # Definimos las frecuencias centrales de las bandas críticas
    cb_c = np.array([50,150,250,350,450,570,700,840,1000,1175,1370,1600,1850,
             2150,2500,2900,3400,4000,4800,5800,7000,8500,10500,13500,19500])  

    cb_c_bark = ftobark(cb_c) # Calculamos la equivalencia a bark de las frecuencias centrales de las bandas críticas    
    cb_in = np.divide(np.multiply(cb_c,N//2),(fs/2))  # Normalizamos para obtener los índices equivalentes (entre 0-256)
    cb_in = cb_in.astype(int) 

    bnd = np.array([0,1,2,3,5,6,8,9,11,13,15,17,20,23,27,32,37,45,52,62,74,88,108,132,180,232]) #Límites de las bandas criticas (entre 100)
    bark = np.arange(25)

    bandwidth_hz = np.array([0,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,22050]) # Ancho de las bandas críticas



    """ Paso 1: Estimamos la densidad espectral de potencia y normalizamos para obtener la equivalencia en SPL."""

    
    audio = audio/(2**(b-1)) # normalizamos la amplitud de la señal conforme al número de bits considerado
        
        
    h = np.hanning(M=512)
    h= h /(np.sqrt(np.sum(np.square(h)/N))) # Normalizamos la ventana por su valor de RMS 
    
    
    X = np.fft.fft(h*audio,N) # Enventanmos la trama y calculamos la fft
    
    fft = abs(X)
    fft = np.square(fft[0:(N//2)]) /N # Calculamos el periodograma
        
    p = 10*np.log10(fft)  # Pasamos a unidades logarítmicas
    delta = 96 - 27.09 # Normalizamos la señal cosiderando que el rango máximo de audición (96 dB) está representado entre los valores lineales ente -1 y 1, y consideramos que el valor de pico (96 dB) corresponde a una trama de todo 1 (10*log10(512)=27.09dB)  
    p += delta
    

        
    
    """ Paso 2: Identificamos las componentes tonales y de banda ancha enmascaradoras"""
    
    """ Calculamos los candidatos a picos enmascaradores"""
    p_tm = []  # Variable para guardar la amplitud de lo que consideramos pico
    k_tm = []  #  Variable oara guardar el índice frecuencial de los picos
    # Buscamos los picos del espectro
    
    for k in np.arange(2,250):
        if (p[k-1] < p[k] and p[k] > p[k+1]):
            
            del_k = []
            
            if k > 2 and k < 63 :
                del_k = np.array([-2,+2])
            elif k >= 63 and k < 127 :
                del_k = np.array([-3,-2,2,3])
            elif k>=127 and k<=256 :
                del_k = np.array([-6, -5, -4, -3, -2, +2, +3, +4, +5, +6])
            else:
                del_k = 0
            
            
            if all(p[k]>p[k+del_k]+7):
                
                p_tm.append(10*np.log10(10**(0.1*p[k-1])+10**(0.1*p[k])+10**(0.1*p[k+1])))
                k_tm.append(k)
                
                #
        k_tm_f = np.multiply(k_tm,22050/256)  # Índices de las componentes enmascaradoras en Hz
        k_tm_f_bark = ftobark(k_tm_f)              # Índices de las componentes enmascaradoras Bark
            
                

                       
    """ Calculamos las componentes no tonales enmascaradoras en las bandas críticas """ 
    
    # Excluimos las componentes ya seleccionadas como tonales
    
    
    idxx = [] # índices de las comoponetes enmascaradoras de banda ancha
    idxe = [] # índices que deberían ser excluidos
    p_nm = [] # valores del ruido enmascarador    
    
    for x in range(0,len(bnd)-1):            # Recorremos todas las bandas críticas
    
        for idx in range(bnd[x], bnd[x+1]):  # Recorremos los ínidices de cada banda crítica
            del_k = []
        
            if idx > 2 and idx < 63 :
                del_k = np.array([-2,-1,0,1,2])
            elif idx >= 63 and idx < 127 :
                del_k = np.array([-3,-2,-1,0,1,2,3])
            elif idx>=127 and idx<=256 :
                del_k = np.array([-6, -5, -4, -3, -2,-1,0,1,2, +3, +4, +5, +6])
            
            for j in range(0,len(k_tm)):  
                for f in range(0,len(del_k)):
                              
                    if (idx != (k_tm[j]+del_k[f])):  # Si el índice no está entre los seleccionados como componentes tonales
                                  
                        c   = "no es necesario"   # No hacemos nada                                            
                        
                    elif (idx == (k_tm[j]+del_k[f])):     # Si el índice está entre los seleccionados como componentes tonales
                        
                        idxe.append(idx)     # Lo excluímos
                                    
    bnd_k = np.arange(0,232)    
    
    # Calculamos los índices excluyendo los seleccionados anterioremente como tonales
    idxx = list(set(bnd_k)^set(idxe))   
            
    for x in range(0,len(bnd)-1):  # Recorremos todas las bandas críticas
        total = 0
        for a in range(len(idxx)):
            
                if (bnd[x]<=idxx[a] and idxx[a]<bnd[x+1]):    
                    
                    total += 10**(0.1*p[idxx[a]])   # Sumamos la potencia de la banda crítica.                                     
        
        p_nm.append(10*np.log10(total))
                        
        
    """Paso 3: Diezmado y Reorganización de componentes enmascarantes."""
    
    p_tm_th = []   # Nivel umbral de enmascarado banda estrecha
    p_nm_th = []  # Nivel umbral de enmascarado banda ancha
    k_nm_th = []  #  Índices umbral de enmascarado banda estrecha
    k_tm_th = []   # Índices umbral de enmascarado banda ancha
    
    for k in range(len(k_tm)):        
                   
        if (p_tm[k] >= th[k_tm[k]]):
            
            p_tm_th.append(p_tm[k])
            k_tm_th.append(k_tm[k])
                
    for l in range(len(cb_in)):        
           
        if (p_nm[l] >= th[cb_in[l]]):
    
             p_nm_th.append(p_nm[l])
             k_nm_th.append(cb_in[l])
        else:
            dummy = l # No hacemos nada
    
    """Enventanamos para suavizar resultados (0.5 Bark Window)."""
        
    k_tm_bark = ftobark(np.multiply(k_tm_th,22050/256))  # convertimos los índices con componentes enmascarantes tonales a bark    
    k_nm_bark = ftobark(np.multiply(k_nm_th,22050/256)) # convertimos los índices con componentes enmascarantes de banda ancha a bark
    
    ptm2 = []
    ktm2 = []
    pnm2 = []
    knm2 = []
    ptm3 = []
    ktm3 = []
    pnm3 = []
    knm3 = []
    
    for k in range(len(k_tm_bark)-1): 
        if np.absolute(k_tm_bark[k+1]-k_tm_bark[k]) <= 0.5 :
            if p_tm_th[k+1] > p_tm_th[k]:
                ptm2.append(p_tm_th[k+1])
                ktm2.append(k_tm_bark[k+1])
            else:
                ptm2.append(p_tm_th[k])
                ktm2.append(k_tm_bark[k])
        else:
            ptm2.append(p_tm_th[k])
            ktm2.append(k_tm_bark[k])
    ptm2.append(p_tm_th[len(k_tm_bark)-1])
    ktm2.append(k_tm_bark[len(k_tm_bark)-1])
            
    for k in range(len(k_nm_bark)-1): 
        if np.absolute(k_nm_bark[k+1]-k_nm_bark[k]) <= 0.5 :
            if p_nm_th[k+1] > p_nm_th[k]:
                pnm2.append(p_nm_th[k+1])
                knm2.append(k_nm_bark[k+1])
            else:
                pnm2.append(p_nm_th[k])
                knm2.append(k_nm_bark[k])            
        else:
            pnm2.append(p_nm_th[k])
            knm2.append(k_nm_bark[k])
    
    pnm2.append(p_nm_th[len(k_nm_bark)-1])
    knm2.append(k_nm_bark[len(k_nm_bark)-1])
      
    
    for k in range(len(ktm2)):
        alone = True
        for j in range(len(knm2)):
            if np.absolute(ktm2[k] - knm2[j]) <= 0.5:
                alone = False
                
                if ptm2[k] < pnm2[j]:
                    pnm3.append(pnm2[j])
                    knm3.append(knm2[j])
                if ptm2[k] >= pnm2[j]:
                    ptm3.append(ptm2[k])
                    ktm3.append(ktm2[k])
        if alone:
            ptm3.append(ptm2[k])
            ktm3.append(ktm2[k])
    
    for k in range(len(knm2)):
        alone = True
        for j in range(len(ktm3)):
            if np.absolute(ktm3[j] - knm2[k]) <= 0.5:
                alone = False
                
        if alone:
            pnm3.append(pnm2[k])
            knm3.append(knm2[k])
    
    """Paso 4: Calculamos los umbrales de enmascaramiento."""
    
    TTMny = []
    TTMty = []
    TTMnx = []
    TTMtx = []
    
    # Aplicamos las funciones de ensanchamiento
    
    def SF(i,j,p):
        
        dz = i-j
        sf = 0
        if -3<=dz<-1:
            sf = 17*dz - 0.4*p+11
        elif -1<=dz<0:
            sf = (0.4*p+6)*dz
        elif 0<=dz<1:
            sf = -17*dz
        elif 1<=dz<8:
            sf = (0.15*p-17)*dz-0.15*p
            
        return sf
        
    for k in range(len(ktm3)):
        j = ktm3[k]
        Ttmy=[]
        Ttmx=[]
        for i in np.arange(j-3,j+8,0.1):
            
            """
            Añadimos los niveles de enmascaramiento tonales
            """
            Ttmy.append( ptm3[k] - 0.275*j -6.025 + SF(i,j,ptm3[k]) ) 
            Ttmx.append( i )         
            
        TTMty.append(Ttmy)
        TTMtx.append(Ttmx)
        
    for k in range(len(knm3)):
        j = knm3[k]
        Ttmy=[]
        Ttmx=[]
        for i in np.arange(j-3,j+8,0.1):
            
            """
            Añadimos los niveles de enmascaramiento banda ancha
            """
            Ttmy.append( pnm3[k] - 0.175*j -2.025 + SF(i,j,pnm3[k]) ) 
            Ttmx.append( i ) 
                    
        TTMny.append(Ttmy)
        TTMnx.append(Ttmx)
    
    
    """Paso 5: Calculamos el nivel de enmascaramiento global."""
    
    g_mth_y = []
    g_mth_x = []
    for i in np.arange(0, 25, 0.1):
        total = 0
        allvals = []
        for j in range(len(freq_bark)):
            if i <= freq_bark[j] < i+0.1:
                
                allvals.append(th[j])
       
        if len(allvals) > 0:
            total += 10 ** (0.1 * np.mean(allvals))
         
        allvals = []
        for n in range(len(TTMnx)):
            
            for k in range(len(TTMnx[n])):
                
                if i <= TTMnx[n][k] < i+0.1:
                    allvals.append(TTMny[n][k])
                    if len(allvals) > 0:
                        total += np.sum(np.power(10, np.multiply(0.1, allvals)))
        allvals = []
        for t in range(len(TTMtx)):
            for k in range(len(TTMtx[t])):
                if i <= TTMtx[t][k] < i+0.1:
                    allvals.append(TTMty[t][k])
                    if len(allvals) > 0:
                        total += np.sum(np.power(10, np.multiply(0.1, allvals)))
        
        if total:
            g_mth_y.append(10 * np.log10(total))
            g_mth_x.append(i)
        
       
    """ Calculamos el umbral por banda y el SMR"""
    g_mth_x=np.array(g_mth_x)
    g_mth_y=np.array(g_mth_y)
    fhz=barktof(np.array(g_mth_x))*N/fs
    umbralporbanda=np.zeros(32)
    umbralauditivo=np.zeros(32)
    banda=int(0)
    for f in range(8,256+8,8):
        fbanda=np.argwhere(fhz<f)
        umbralauditivo[banda]=np.min(th[f-8:f])
        umbralporbanda[banda]=np.max([np.min(g_mth_y[fbanda]),umbralauditivo[banda]])
        banda=banda+1    
    

    SMR=np.zeros(32)
    for banda in range(32):
        SMR[banda]=np.max(p[banda*8:banda*8+8])-umbralporbanda[banda]

    return p,th,umbralporbanda,SMR

def redistribucion_bits_MPEG1(SMR0,R=4,bit_scale=6,bit_banda=4):
    '''redistribucion_bits_MPEG1 calcula como distribuir el número de bits necesarios para cada una de las 32 bandas
    consideradas en la codificación definida en la capa I del MPEG-1 a partir de la relación señal a enmascaramiento 
    en cada banda.
    
    Argumentos de entrada:
    SMR0 (ndarray) -> vector con las relaciones de señal a enmascaramiento para cada una de las 32 bandas calculadas
    según modelo psicoacústico
    R: (escalar) -> Ratio de compresión deseado
    bit_scale (escalar)  -> Número de bits reservados para cuantificar el factor de escalado de cada banda
    bit_banda (escalar)  -> Número de bits reservados para cuantificar el número de bits dedicado a cada banda
    
    Salida:
    bit_por_banda (ndarray) -> vector con el número de bits asignados a cada banda
    SNR (escalar) -> Relación señal a ruido de cuantificación obtenida en cada banda (considerando que en cada banda hay un tono de la misma amplitud que el rango dinámico del cuantificador)
    flag (booleano) -> Indica si se ha podido cumplir el reparto de bits satisfaciendo el enmascaramiento o no (perdiendo calidad) 
    '''
    
    # Despreciamos las últimas 5 bandas imponiendo una SMR muy pequeña
    # Así garantizamos que se dediquen 0 bits a dichas bandas
    
    flag=True  # Inicialmente consideramos que se van a poder distribuir todos los bits (no van a faltar)
    SMR=1*SMR0
    SMR[28:] = -100 
    # Inicializamos los bits disponibles en función del factor de compresión deseado y de los bits de la información lateral necesarios
    bits_disponibles = np.floor(384*(16/R))- bit_scale*27-bit_banda*27
    bits_usados=0; # Inicializamos el número de bits repartidos
    
    # El criterio para repartir los bits es maximizar la relación señal a ruido frente a la de señal a enmascaramiento (MNR=SNR-SMR)
    
    bit_por_bandas = np.zeros(32) # inicializamos cada banda con 0 bits
    SNR = np.zeros(32) # Con 0 bits la SNR inicial es cero en cada banda
    MNR = SNR-SMR


    while bits_usados < bits_disponibles:
        kmin = np.argmin(MNR)
        if bit_por_bandas[kmin] == 16:
            SNR[kmin] = 100 # Evitamos codificar caulquier banda con más de 16 bits
        else:
            if bit_por_bandas[kmin]==0: 
                bit_por_bandas[kmin] = 2 # Primeros bits que se asignan a la banda (evitamos que una banda tenga sólo 1 bit)
            else:
                bit_por_bandas[kmin] = bit_por_bandas[kmin]+ 1 # Incrementamos un bit en esa banda 
        
        # Asumimos una señal sub-banda tonal para recalcular la SNR
        SNR[kmin]=1.77+6.02*bit_por_bandas[kmin]
    
        MNR[kmin] = SNR[kmin] - SMR[kmin]
        bits_usados = 12*np.sum(bit_por_bandas)

    if any(MNR<0):
        flag=False
    
    return bit_por_bandas,SNR,flag

def panning_estereo(theta,theta0,gL=1):
    ''' Función que calcula las ganancias a aplica a un sistema de reproducción estéreo con una separación entre altavoces de 2*theta0 para realizar un Pannig de theta grados
    
    Argumentos de entrada:
    theta: ángulo en grados de la fuente virtual
    theta0: ángulo en grados de la mitad de la separación entre altavoces
    
    Salida:
    gL: ganancia a aplicar al altavoz izquierdo
    gR: gananica a aplicar al altavoz derecho
     
    
    '''
    gL=gL*np.ones(len(np.array([theta])))
    q=np.tan(theta*np.pi/180)/(np.tan(theta0*np.pi/180)+1e-16)
    
    gR=(gL+q)/(gL-q)
    
    # normalizamos amplitudes
    gRn=gR/(np.sqrt(gR**2+gL**2))
    gLn=gL/(np.sqrt(gR**2+gL**2))
    return gLn,gRn

def panning_estereo_tr(x,theta0=30,fs=44100,B=512):
    '''Realizamos panning de amplitud en tiempo real sobre la señal x .

      Argumentos de entrada:
        x: Señal a reproducir 
        theta0: mitad del ángulo de apertura
        fs (escalar): frecuencia de muestreo
        B (escalar): Tamaño del bloque de datos  
        '''
    
    global CHUNK,xss,n,numtramas,theta0g
    
    CHUNK=1*B 
    theta0g=1*theta0
    numtramas=np.ceil(len(x)/CHUNK)
    xss=np.zeros(int(numtramas)*CHUNK)
    xss[0:len(x)]=x

  
    n=0
    
                 

    

    # Botones y controles
    botonDet=widgets.Button(
            description='Detener',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    botonEmp=widgets.Button(
            description='Empezar',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    style = {'description_width': 'initial'}
    barra=widgets.FloatSlider(
    value=0,
    min=-theta0,
    max=theta0,
    step=0.1,
    description='Ángulo de la fuente',
    style=style,
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

    
        # MOSAICO VISUALIZACIÓN
    display(widgets.VBox([barra,widgets.HBox([botonEmp, botonDet])]))
    
    # Instancia a clase pyaudio
    p = pyaudio.PyAudio()
    
    def callback(in_data, frame_count, time_info, status):
        global CHUNK,xss,n,numtramas,theta0g
        flag= pyaudio.paContinue
        
        
        if n==numtramas*CHUNK:
            n=0
        in_data=xss[n:n+CHUNK]
        angulo=barra.value
        
        gL,gR=panning_estereo(angulo,theta0g)
        in_data_estereo=np.zeros([CHUNK,2])
        in_data_estereo[:,0]=gL*in_data
        in_data_estereo[:,1]=gR*in_data
        n=n+CHUNK
           
        return (bytes(in_data_estereo.astype(np.int16)),flag)      
        
        
    stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=fs,
                input=False,
                output=True,
                start=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)    
    
    def on_button_clickedDet(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True
        

    def on_button_clickedEmp(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True       

        
        
    botonDet.on_click(on_button_clickedDet)
    botonEmp.on_click(on_button_clickedEmp)
    
    
    
def simpleHRIR(theta, fs=44100):
    '''Función que devuelve un filtro FIR sencillo como HRIR para el ángulo de acimut y frecuencia de muestreo introducidas como parámetro de entrada
    
    Argumentos de entrada:
    theta (escalar) ángulo de azimuth en grados
    fs (escalar) frecuencia de muestreo en Hercios
    
    Argumentos de salida:
    hrir (ndarray): Vector con la respuesta al impulso de la HRIR
    '''
    theta = theta + 90
    theta0 = 150 
    alfa_min = 0.05 
    c = 334 # Velocidad del sonido
    a = 0.08 # Radio de la cabeza
    w0 = c/a
    delta=np.zeros(int(np.rint(0.003*fs))) 
    delta[0]=1
    alfa = 1+ alfa_min/2 + (1- alfa_min/2)* np.cos(theta/ theta0* np.pi) 
    # Numerador de la función de transferencia
    b = np.array([(alfa+w0/fs)/(1+w0/fs), (-alfa+w0/fs)/(1+w0/fs)]) 
    # Denominador de la función de transferencia
    a = [1, -(1-w0/fs)/(1+w0/fs)] 

    if (np.abs(theta) < 90):
        gdelay = np.rint(- fs/w0*(np.cos(theta*np.pi/180) - 1)) 
    else:
        gdelay = np.rint(fs/w0*((abs(theta) - 90)*np.pi/180 + 1) )
    

    out_magn = signal.lfilter(b, a, delta)
    if gdelay>0:
        hrir = np.append(np.zeros(int(gdelay)), out_magn[:-int(gdelay)])
    else:
        hrir = out_magn
    
    return hrir



def HRIR_tr(x,fs=44100,B=512):
    '''Realizamos virtualización de la posición en acimut y distancia mediante modelo simple de HRIR .

      Argumentos de entrada:
        x: Señal a reproducir 
        fs (escalar): frecuencia de muestreo
        B (escalar): Tamaño del bloque de datos  
        '''
    
    global CHUNK,xss,n,numtramas,fsg,z0l,z0r
    
    CHUNK=1*B 
    numtramas=np.ceil(len(x)/CHUNK)
    xss=np.zeros(int(numtramas)*CHUNK)
    xss[0:len(x)]=x
    fsg=1*fs
  
    n=0
    
    h0=simpleHRIR(0, fs=fsg)
    z0l=np.zeros(len(h0)-1)
    z0r=np.zeros(len(h0)-1)
                 

    

    # Botones y controles
    botonDet=widgets.Button(
            description='Detener',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    botonEmp=widgets.Button(
            description='Empezar',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    style = {'description_width': 'initial'}

    barratheta=widgets.IntSlider(
    value=0,
    min=-180,
    max=180,
    step=1,
    style =style,
    description='Ángulo de la fuente',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

    barrar=widgets.FloatSlider(
    value=1,
    min=1,
    max=10,
    step=0.2,
    style =style,
    description='Distancia de la fuente',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

    
        # MOSAICO VISUALIZACIÓN
    display(widgets.VBox([barratheta, barrar,widgets.HBox([botonEmp, botonDet])]))
    
    # Instancia a clase pyaudio
    p = pyaudio.PyAudio()
    
    def callback(in_data, frame_count, time_info, status):
        global CHUNK,xss,n,numtramas,fsg,z0r,z0l
        flag= pyaudio.paContinue
        
        
        if n==numtramas*CHUNK:
            n=0
        in_data=xss[n:n+CHUNK]
        ang=barratheta.value
        distancia=barrar.value
        
        # Dividimos la respustas de los filtros por la mitad para evitar saturar
        hrirl=0.5*(1/distancia)*simpleHRIR(ang, fs=fsg)
        hrirr=0.5*(1/distancia)*simpleHRIR(-ang, fs=fsg)
        
        in_datal,z0l=signal.lfilter(hrirl,1,in_data,zi=z0l)
        in_datar,z0r=signal.lfilter(hrirr,1,in_data,zi=z0r)
        
        
        in_data_estereo=np.zeros([CHUNK,2])
        in_data_estereo[:,0]=in_datal
        in_data_estereo[:,1]=in_datar
        n=n+CHUNK
           
        return (bytes(in_data_estereo.astype(np.int16)),flag)      
        
        
    stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=fs,
                input=False,
                output=True,
                start=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)    
    
    def on_button_clickedDet(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True
        

    def on_button_clickedEmp(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True       

        
        
    botonDet.on_click(on_button_clickedDet)
    botonEmp.on_click(on_button_clickedEmp)
    
    
def addclick(x,Minnumclick=2,Maxnumclick=10,MinDurclick=10,MaxDurclick=36):
    '''Función que añade degradaciones localizadas a la señal de audio introducida como entrada.
    Las degradaciones se generan de duración aleatoria y de en instantes aleatorios dentro de la señal original.
    
    Argumentos de entrada
    x -> ndarray con la señal de audio
    Minnumclick (int) -> Número de tramos defectuosos mínimos deseados (2, por defecto)
    Maxnumclick (int) -> Número de tramos defectuosos máximos deseados (10, por defecto)
    MinDurclick (int) -> Duración mínima en muestras de las degradaciones localizadas (10, por defecto)
    MaxDurclick (int) -> Duración máxima en muestras de las degradaciones localizadas (36, por defecto)
    
    Variables de salida:
    y -> ndarray de las mismas dimensiones que x pero con la señal degradada
    '''
    
    numclick=int(Minnumclick+np.rint((Maxnumclick-Minnumclick)*np.random.rand(1)))
    

    Tamsignal=len(x);
    y=x*1
    Muesini=10*MaxDurclick
    Muesfin=Tamsignal-10*MaxDurclick

    iniclick=np.sort(Muesini+np.rint((Muesfin-Muesini)*np.random.rand(numclick)))
    durclick=MinDurclick+np.rint((MaxDurclick-MinDurclick)*np.random.rand(numclick))
    
    for click in range(numclick):
        sclick=np.random.randn(int(durclick[click]))
        sclick=(0.8+0.15*np.random.rand(1))*(2**15-1)*sclick/np.max(np.abs(sclick))
        y[int(iniclick[click]):int(iniclick[click]+durclick[click])]=sclick
    
    return y


def restaura(x,fs=8000, N='none',Dur=36, tr=0.05, ta=0.0001, Ttrama=0.030, Pumbral=1, preataque=2, metodo=1):
    '''Función que añade restaura una señal de audio degenerada con degradaciones localizadas.
    
    Argumentos de entrada
    x -> ndarray con la señal de audio
    fs (float) frecuencia de muestreo (8000 Hz por defecto)
    N (int) -> Orden de los filtros de predicción lineal usados para regenerar la señal ('none', por defecto para calcular el orden de forma automática)
    Dur (int) -> Número de muestras que se ausume para la duración de las degradaciones 
    tr (float) -> Tiempo de integración para el cálculo de la envolvente (0.05 segundos por defecto)
    ta (float) -> Tiempo de integración de ataque para el cálculo de la envolvente (0.0001 segundos por defecto)
    Ttrama=0.030 -> Tiempo de la trama anterior al defecto para la obtención de los coeficientes de predicción lineal (30ms por defecto)
    Pumbral (float) -> Poderación del umbral empleado para detectar las degradaciones (1, por defecto)
    preataque (int) -> número de muestras previas a la detección del defecto desde las que iniciamos la recontrucción (2, por defecto)
    metodo (int) ->1 filtrado LPC, 2-> repetición periódica del último ciclo
    
    Variables de salida:
    y -> ndarray de las mismas dimensiones que x pero con la señal restaurada    '''
    
    
    L=int(Ttrama*fs)
    # Obtenemos los inicios y finales de las degradaciones
    
    yd= envolvente(x, tr=0.05 , fs=fs, ta=0.0001)
    dy=np.diff(yd)
    ymed=signal.medfilt(np.abs(x), 2*Dur+1)
    umbral = Pumbral*np.max(ymed)
    inicios=np.argwhere(dy>umbral)  # Seleccionamos, en primera instancia, los valores que superan el umbral, como inicios de las degradaciones
    inicios=inicios-preataque # desplazamos los inicios unas muestras de seguridad

    mascara=np.zeros(len(x)) # creamos una mascara binaria (inicializada a ceros) para indicar los ínidices donde hay degradaciones
    
    for inicio in inicios:
        mascara[int(inicio):int(inicio+Dur)]=np.ones(Dur)  # Fijamos a unos las muestras a partir de los inicios (usamos la duración del modelo usado de degradación)

    # Seleccionamos las muestras iniciales y finales definitivas de las degradaciones:
    inicios=1+np.argwhere(np.diff(mascara)==1)
    finales=inicios+Dur
    Tini=int(np.floor(fs/300)) # Número de muestas mínimo del periodo del Pitch considerado
    Tfin=int(np.ceil(fs/60)) # Número de muestas máximo del periodo del Pitch considerado
    # Restauramos cada trozo de la señal degradado
    y0=x*1
    
    for tramo_deg in range(len(inicios)):
        ini=int(inicios[tramo_deg])
        trama=x[ini-L:ini] # Obtenemos una trama de datos válida previa al defecto
        if metodo==1:
            if N=='none': #Calculamos automáticamente el orden del filtro LP en función del pitch de la señal
                N0=2+int(np.ceil(fs/1000))
                a,Pe= lpc(trama,N0) #Calculamos coeficientes del filtro de análisis LPC para calcular el Pitch
                error=signal.lfilter(a,1,trama)  # calculamos el error de predicción
                Cx=signal.correlate(error,error)/len(error) #calculamos la autocorrelación del error de predicción
                # Nos quedamos con las muestras que representan valores de pitch válidos 
                Cxx=Cx[len(error)-1:len(error)-1+Tfin]
                Cxx[0:Tini]=0
                Imax=np.argmax(Cxx) # Obtenemos el número de muestras del periodo del pitch
        
                N=Imax
                while Dur>N:
                    N+=N
        
            a,Pe= lpc(trama,N) #Calculamos coeficientes del filtro de análisis LPC
            coeflpc=np.concatenate((np.array([0]),-a[1:]))  # obtenemos el filtro de predicción lineal
            # Restauramos la señal
            estado=1*x[ini-1:ini-N-1:-1]  # Definimos el buffer de datos inicial con las N muesrtas anteriores al defecto
        
            # Filtramos muestra a muestra (calculamos las muestras filtradas en el tramo en que las muestras de entrada son ceros)
            for muestradef in range(Dur):  
                y0[ini+muestradef]=np.sum(coeflpc[1:]*estado)
                estado=np.concatenate(([y0[ini+muestradef]], estado[:-1]))       # Actualizamos el buffer de los datos a filtrar 
        elif metodo==2:
            N0=2+int(np.ceil(fs/1000))
            a,Pe= lpc(trama,N0) #Calculamos coeficientes del filtro de análisis LPC para calcular el Pitch
            error=signal.lfilter(a,1,trama)  # calculamos el error de predicción
            Cx=signal.correlate(error,error)/len(error) #calculamos la autocorrelación del error de predicción
            # Nos quedamos con las muestras que representan valores de pitch válidos 
            Cxx=Cx[len(error)-1:len(error)-1+Tfin]
            Cxx[0:Tini]=0
            Imax=np.argmax(Cxx) # Obtenemos el número de muestras del periodo del pitch
            
            #Nos centramos en una de las degradaciones y seleccionamos el último periodo de señal válido antes del defecto:
            periodo=1*x[ini-Imax:ini]

            # Completamos las muestras del defecto con muestras del periodo obtenido de forma cíclica
            k2=0
            for k in range(Dur):
                if k2 == Imax:
                    k2=0
                y0[ini+k]=periodo[k2]
                k2 += 1
                

    return y0


def denoiser(x,fs,Tb,Tha,huella,peso=1,delta=1e-10,ventana='boxcar',c=1e-10):
    '''Algoritmo que elimina el ruido usando supresión espectral a partir de una huella de ruido conocida
    
    Argumentos de entrada
    x -> ndarray con la señal de audio
    fs (float) frecuencia de muestreo 
    Tb (float) -> Tiempo en segundos para la ventana de análisis en el procesado por bloques
    Tha (float) -> Tiempo de salto entre procesados 
    huella (ndarray) -> Vector de ceil(fs*Tb) con la densidad espectral de la huella de ruido
    peso (int) -> Peso a aplicar a la huella de ruido (1 por defecto)
    ventana (str)-> Cadena con la ventana usada en el procesado por bloques
    c (float) -> constante para aplicar efecto de denoiser musical sobre la señal antes de aplicar la supresión espectral
    
    
    Variables de salida:
    out -> ndarray de las mismas dimensiones que x pero con la señal con la supresión de ruido    '''
    
    
    
    x=x.astype(np.float64)
    Nb=int(np.ceil(fs*Tb))
    Nha=int(np.ceil(fs*Tha))
    
    
    # Creación ventana (por defecto, rectangular)
    w=signal.get_window(ventana,Nb)
    # Criterio COLA
    K=sum(w**2)/Nha
    w=w/np.sqrt(K)    
    
    xin=np.concatenate([np.zeros(Nb),x,np.zeros(Nb)]) # Añadimos un bloque de información al principio y al final para evitar el efecto transitorio debido al enventanado
    Numbloques=1+(np.ceil((len(xin)-Nb)/Nha)); # Calculamos el número de bloques que debemos procesar
 
    out=np.zeros(int(Numbloques*Nha+Nb)); #Inicializamos la salida del procesado a ceros

    # Incializamos los índices que recorren la señal para su análisis y reconstrucción
    p_in=0; # Puntero que indica la posición de lectura en cada trama para el análisis
    


    # Recorremos la señal bloque a bloque para realizar el análisis-procesado-síntesis 

    while p_in< len(xin)-Nb:  
        bloque=xin[p_in:p_in+Nb];   # Adquirimos el bloque a procesar
    
        ##############
        # ANÁLISIS
        bloque=bloque*w      # Enventanamos el bloque
        fftbloque=np.fft.fft(np.fft.fftshift(bloque));    # Calculamos la fft del bloque al que le hemos aplicado un retardo circular para centrarlo en el máximo de la ventana (operación similar a si usamos un banco de filtros de ancho de banda el de un bin frecuencial y fase cero)
    
        # PROCESADO
        modulo_in=np.abs(fftbloque)
        fftbloque=fftbloque*modulo_in/(modulo_in+c);
        H=np.maximum(modulo_in-peso*huella,0)/(modulo_in+delta)
    
        # Procesamos el módulo y la fase para aplicarle la transformación a la señal (en este caso no le hacemos nada)
        bloque_out=np.real(np.fft.ifft(H*fftbloque))
        
        #SÍNTESIS
        # Resintetizamos la señal
              
        bloque_out=np.fft.ifftshift(bloque_out)*w;    # Deshacemos el retardo circular y resintetizamos enventanando.
        
        ###########################################################
    
        # Almacenamos el bloque procesado en la señal de salida 
        out[p_in:p_in+Nb]=out[p_in:p_in+Nb]+bloque_out;
    
        # Actualizamos los punteros para el siguiente bloque de análisis-síntesis
    
        p_in=p_in+Nha;
        
    
        #Fin del procesado por bloques
    return out[Nb::]   # Descartamos el primer bloque para sincronizar la señal de salida con la de entrada (se añadió ficticiamente al principio para evitar transitorios)



def pitch_mark(x,f0,fs,Nhop):
    m=[]
    for i in range(len(f0)):
        
        if (i==0 and f0[i]==0):
            f0[i] = 120; # 120Hz en el caso de no existir un pitch previo
        elif (f0[i]==0):
            f0[i] = f0[i-1];

        P0 = int(np.rint(fs/f0[i])); # Periodo fundamental de la trama i-ésima

        if i==0:
            marca=np.argmax(x[0:P0])  # Primera marca
            m=np.append(m,marca)
            marca=np.arange(int(m[-1]),int((i+1)*Nhop),P0)
            m=np.append(m,marca[1:])    # Resto de marcas de la primera trama
        marca=np.arange(int(m[-1]),int((i+1)*Nhop),P0)
        m=np.append(m,marca[1:])    # Resto de marcas del resto de tramas
    return m        
        

def psolaf(x,m,alpha,beta,gamma):
    '''
    x -> señal de entrada
    m  -> marcas del pitch 
    alpha -> factor de expansión/compresión temporal
    beta -> factor de desplazamiento entre marcas (por si no se desea recorrerlas respetando las marca originales) 
    gamma -> factor de expansión/compresión frecuencial  '''
    
    P = np.diff(m); #calcula periodos de los pitch
    if m[0]<=P[0]:  #eliminamos la primera marca del pitch si está antes del primer periodo completo
        m=m[1:len(m)];
        P=P[1:len(P)];
    
    if m[-1]+P[-1]>len(x): # elimina la última marca pitch si conduce a buscar muestras en un rango mayor que la duración de la señal
        m=m[0:-1];
    else:
        P=np.append(P, P[-1]); # o repetimos el último periodo de Pitch en el tramo de la última marca hasta el final
    
    Lout=int(np.ceil(len(x)*alpha));
    out=np.zeros(Lout); # señal de salida
    tk = P[0] # marca del pitch
    while np.rint(tk)<Lout:
        i = np.argmin( np.abs(alpha*m - tk) ); #buscamos el segmento de análisis desplazado más cercano a la marca del pitch correspondiente
        pit=P[i];  # Obtenemos el pitch de dicho segmento
        pitstr=pit/gamma  # Calculamos el Pitch deseado
        gr = x[int(m[i]-pit):int(m[i]+pit+1)] # Cogemos dos periodos de la señal centrados en la marca temporal
        gr = gr * np.hanning(len(gr)) # Enventanamos dicho par de periodos
        
        grstr= signal.resample(gr,int(len(gr)/gamma)) 
        
        # Buscamos la posición inicial que le correspondería a la señal reconstruida con el deplazamiento adecuado
        iniGr=int(np.rint(tk)-pitstr);  
        #print(np.shape(grstr),iniGr,Lout)
        if iniGr>0 and iniGr+len(grstr)<Lout:
            out[iniGr:iniGr+len(grstr)] = out[iniGr:iniGr+len(grstr)]+grstr; # solapamos los nuevos segmentos
            
        tk=tk+pit/beta;  # Desplazamos la marca temporal una distancia proporcinal al pitch considerado para continuar con la reconstrucción
    
        
    return out

def specenv_win(x,ventana='hann',nob=11,fs=44100):
    """ Calcula la respuesta en frecuencia y la envolvente frecuencial del la señal de entrada
     
     Argumentos de Entrada: 
     x (ndarray): vector con la señal de entrada
     ventana: string con el timpo de enventanado 
     nob: número de bins de promediado para el cálculo de la envovente espectral
     fs: frecuencia de muestreo
     
     Salidas:
     fa (ndarray): vector con las frecuencias analógicas donde se ha calculado la información
     flog(ndarray): respuesta en frecuencia en unidades logarítmicas
     flog_rms(ndarray): envolvente espectral en unidades logarítmicas
    """

    
    
    WLen=len(x)
    WLen=int(np.max([1,WLen-1+WLen%2]))  #forzamos a que sea impar
    x=x[:WLen]
    # Ancho de banda del filtro de promediado: nob*fs/WLen
    
    w=signal.get_window(ventana,int(WLen))
    X=np.fft.fft(x*w)/(WLen/2)
    fa=fs*np.arange(WLen)/WLen
    
    fmax=int(1+np.floor(WLen/2))
    
    nob=int(np.max([1,nob+1-nob%2]))  #forzamos a que sea impar
    vent1=signal.get_window(ventana,int(nob))
    f_channel=np.concatenate([np.zeros(int((WLen-nob)/2)),vent1,np.zeros(int((WLen-nob)/2))])
    
    fft_channel=np.fft.fft(np.fft.fftshift(f_channel))
    
    X2=X*np.conj(X)
    energia=np.real(np.fft.ifft(np.fft.fft(X2)*fft_channel))
    flog_rms=10*np.log10(np.abs(energia))
    flog=20*np.log10(np.abs(X))
    
    
    
    return fa[:fmax],flog[:fmax], flog_rms[:fmax]

def convolucion_interactiva(x1,h,n):
    y=np.zeros(2*(len(x1)+len(h))+1)
    centro=int(((len(x1)+len(h))))
    conv=signal.convolve(x1,h)
    y[centro:centro+len(x1)+len(h)-1]=conv
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.subplots_adjust(hspace=0.55)
    plt.stem(np.arange(len(x1)),x1)
    plt.stem(np.arange(len(h))+n-len(h)+1,h[-1::-1],linefmt='r',markerfmt='or',use_line_collection=True)
    plt.xlim(-len(x1)-len(h)-5,len(x1)+len(h)+5)
    cad2='h({mt}-k)'
    plt.legend(['x(k)', cad2.format(mt=n)],loc="upper left")
    plt.subplot(212)
    font = {'color':  'green'}
    plt.stem(np.arange(-(len(x1)+len(h)),len(x1)+len(h)+1),y,linefmt='g',markerfmt='og',use_line_collection=True)
    plt.plot(np.array(n),np.array(y[n+len(x1)+len(h)]),'ko')
    plt.title('y(n)=x(n)*h(n)',fontdict=font)
    plt.legend(['y({mt})={convt}'.format(mt=n,convt=y[n+len(x1)+len(h)])],loc="upper left")
    
    plt.xlim(-len(x1)-len(h)-5,len(x1)+len(h)+5);
    
def replica(x,N,ini,fin):
    
    n=np.arange(ini,fin)
    ini0=ini-ini%N
    fin0=fin+fin%N-1
    n0=ini-ini0
    
    y=np.zeros(fin0-ini0+1)
    
    x_ext=np.zeros(N)
    x_ext[0:int(np.min([N,len(x)]))]=x[0:int(np.min([N,len(x)]))]
        
    for periodos in range(len(y)//N):
        y[0+periodos*N:N+periodos*N]=x_ext
    y=y[n0:n0+fin-ini]
    
    
    return y,n

def redmodN(x,N):
    y=1*x;
    if N>len(x):
        y=np.concatenate([x,np.zeros(N-len(x))])
    elif N>0:
        NumN=int(np.ceil(len(x)/N))
        if NumN>1:
            x_ex=np.zeros(NumN*N)
            x_ex[0:len(x)]=x
            y=np.zeros(N)
            for k in range(NumN):
                y += x_ex[0+k*N:N+k*N]
    
    return y

def redmodN_interactiva(x,N):
    y=redmodN(x,N)
    plt.stem(y)
    plt.xlim(-1,len(x)+1)
    plt.title('Reducción modulo {Nt} de la secuencia x(n)'.format(Nt=N))

def retcircular_interactiva(x1,N,n):
    xp=redmodN(x1,N)
    y,n2=replica(xp,N,-int(len(x1)*4),int(len(x1)*4))
    n2=n2+n
    cero=int(np.argwhere(n2==0))
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.subplots_adjust(hspace=0.55)
    plt.stem(np.arange(len(xp)),y[cero:cero+N])
    plt.xlim(-len(x1)*2.5,len(x1)*2.5)
    cad2='x((n-{nt}))_{Nt}'
    plt.title('Señal de duración limitada: x((n-({nt})))_{Nt}'.format(nt=n,Nt=N))
    plt.subplot(212)
    plt.stem(n2,y)
    plt.axvspan(0-0.5,N-1+0.5,color='orange',alpha=0.3);
    plt.title('Señal periódica: x(n-({nt}))'.format(nt=n))
    plt.xlim(-len(x1)*2.5,len(x1)*2.5);
    
def convcircular_interactiva(x1,h,N,n):
    
    conv=redmodN(signal.convolve(redmodN(x1,N),redmodN(h,N)),N)
    h_ext=redmodN(h,N)
    h_ext=h_ext[-1::-1]
    
    
    if n>N-1:
        n=N-1
        cad1='n debe ser menor que N'
    else:
        cad1=''
    D=n%N
    h_ext=np.concatenate([h_ext[-1-D:],h_ext[0:-1-D]])
    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.subplots_adjust(hspace=0.75)
    plt.stem(np.arange(N),redmodN(x1,N))
    plt.stem(np.arange(N),h_ext,linefmt='r',markerfmt='or',use_line_collection=True)
    cad2='h(({mt}-k))'
    plt.legend(['x(k)', cad2.format(mt=n)],loc="upper right")
    plt.title(cad1)
    plt.xlim(-1,len(x1)+len(h)+3);
    plt.subplot(312)
    font = {'color':  'green'}
    plt.stem(np.arange(N),conv,linefmt='g',markerfmt='og',use_line_collection=True)
    plt.title('Convolución circular',fontdict=font)
    plt.plot(np.array(n),np.array(conv[n]),'ko')
    plt.legend(['y({mt})={convt}'.format(mt=n,convt=conv[n])],loc="upper right")
    plt.xlim(-1,len(x1)+len(h)+3);
    plt.subplot(313)
    plt.stem(np.arange(len(x1)+len(h)-1),np.convolve(x1,h),linefmt='k',markerfmt='ok',use_line_collection=True)
    plt.title('Convolución Lineal')
    plt.xlim(-1,len(x1)+len(h)+3);
           
def convcircvlin_interactiva(x,h,N):
    y=signal.convolve(x,h)
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.subplots_adjust(hspace=0.55)
    plt.stem(np.arange(len(y)),y)
    plt.xlim(-1,len(y)+15)
    plt.title('Convolución lineal')
    
    y_c=redmodN(y,N)
    N2=len(y)-N;
    
    plt.subplot(212)
    if N2<0:
        plt.stem(np.arange(N),y_c,linefmt='g',markerfmt='og',use_line_collection=True)
    else:
        if (N2>0 and N2<N):
            plt.stem(np.arange(N2),y_c[0:N2],linefmt='r',markerfmt='or',use_line_collection=True)
        if N2<N:
            plt.stem(np.arange(N2,N),y_c[N2:N],linefmt='g',markerfmt='og',use_line_collection=True)
        else:
            plt.stem(np.arange(N),y_c,linefmt='r',markerfmt='or',use_line_collection=True)
    plt.title('Convolución cicular')
    if N2<=0:
        plt.legend(['Valores válidos de la convolución lineal'])
    else:
        plt.legend(['Valores no válidos de la convolución lineal', 'Valores válidos de la convolución lineal'])
    plt.xlim(-1,len(y)+15);

def pitchshifting_tr(fs,Tb,Tdg=0.03,x=np.array([None,None])):
    '''Realiza en tiempo real el efecto de pitch shifting.

      Argumentos de entrada:
        fs (escalar): frecuencia de muestreo
        Tb (escalar): Duración del bloque de datos
        Td (escalar): Máximo retardo retardo. Por defecto 30ms 
        x: Señal a representar o NONE para analizar la señal captada por el micrófono'''
    
    
    global In,fss,CHUNK,xss,Nbuf,m,Td,n,numtramas,Nb,ph1,ph2,Overlap,fgain,sd,Ret1,Ret2
    
    Ret1=0
    Ret2=0
    B=int(np.ceil(fs*Tb))
    Td=1*Tdg
    Overlap=0.1  # Porcentaje de solapamiemto entre las dos líneas de retardo (10%)
    fgain = 1 / Overlap
    Nb=int(np.ceil(fs*Tb))  # Muestras del bloque
    sd=int(np.ceil(fs*Td))  # Muestras del retardo 

    Nbuf=np.zeros(2+Nb+sd)  # Buffer de datos incluyendo líneas de retardo

    CHUNK=1*B 
    
    ph1 = 0                 # fases normalizadas de las funciones de las líneas de retardo
    ph2 = (1 - Overlap)


    In=False
   
    if x[0]==None:
        In=True
    else:
        numtramas=np.ceil(len(x)/CHUNK)
        xss=np.zeros(int(numtramas)*CHUNK)
        xss[0:len(x)]=x

  
    n=0

                 
    
    fss=1*fs

    # Botones y controles
    botonDet=widgets.Button(
            description='Detener',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    botonEmp=widgets.Button(
            description='Empezar',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    
    barraFactor=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=-12,
            max=12,
            step=0.5,
            description='Factor de desplazaminto del Pitch',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=widgets.Layout(width='70%', height='100px'),
            )

        # MOSAICO VISUALIZACIÓN
    display(widgets.HBox([botonEmp, botonDet, barraFactor]))
    
    # Instancia a clase pyaudio
    p = pyaudio.PyAudio()
    
    
    def callback(in_data, frame_count, time_info, status):
        global In,fss,CHUNK,xss,Nbuf,m,Td,n,numtramas,Nb,ph1,ph2,Overlap,fgain,sd,Ret1,Ret2
        flag= pyaudio.paContinue
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        yb=np.zeros(CHUNK)
        m = (1-2**(barraFactor.value/12)) # Variación del retardo (en muestras) de una muesra a la siguiente
        pRate = m / Td          # Variación del retardo en muestras por segundo
        pstep = pRate / fss      # Paso de las fases (adimensional)

        if In==False:
            if n==numtramas*CHUNK:
                n=0
            in_data=xss[n:n+CHUNK]
            n=n+CHUNK
        else:
            in_data=audio_data
        
         
        Nbuf=np.concatenate((in_data[-1::-1],Nbuf[:-Nb]))
        if m==0:
            out_data=1*in_data
            
        else:
            # Procesamos muestra a muestra dentro de cada bloque
            
            for muestra in range(Nb):
                ph1 = (ph1 + pstep)%1
                ph2 = (ph2 + pstep)%1
                
            # línea de retardo 2 se acerca a su fin. Comenzamos solape con linea de retardo 1
                if ((ph1 < Overlap) and (ph2 >= (1 - Overlap))):
                    Ret1 = sd * ph1 
                    Ret2 = sd * ph2 
                    
                    Gan1  = np.cos((1 - (ph1* fgain)) * np.pi/2)
                    Gan2  = np.cos(((ph2 - (1 - Overlap)) * fgain) * np.pi/2)
                    
            # Línea de retardo 1 está activa
                elif ((ph1 > Overlap) and (ph1 < (1 - Overlap))):
                    # El retardo de la línea 2 se matiene fijo mientras la línea 1 está activa
                    ph2 = 0.0
                    Ret1 = sd * ph1 
             
                    Gan1 = 1.0
                    Gan2 = 0.0
                   
            # Linea de retardo 1 se acerca a su fin. Comenzamos solape con linea de retardo 2
                elif ((ph1 >= (1 - Overlap)) and (ph2 < Overlap)):
                    Ret1 = sd * ph1
                    Ret2 = sd * ph2 
                    
                    Gan1 = np.cos(((ph1 - (1 - Overlap)) * fgain) * np.pi/2)
                    Gan2 = np.cos((1 - (ph2* fgain)) * np.pi/2)
                    
            # Linea de retardo 2 está activa
                elif((ph2 > Overlap) and (ph2 < (1 - Overlap))):
                    # El retardo de la línea 1 se matiene fijo mientras la línea 2 está activa
                    ph1 = 0.0    
                    Ret2 = sd * ph2 
                    
                    Gan1 = 0.0
                    Gan2 = 1.0
                    
            
            # Aplicamos las líneas de retardo
                
            #Linea 1
                  
                retardo1 = Nb-muestra + Ret1  
                Nretardo1=np.floor(retardo1)      # Redondeo del retardo a nº de muestras enteras.
                frac1=retardo1-Nretardo1           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
                y1=Nbuf[int(Nretardo1)]*(1-frac1)+Nbuf[int(Nretardo1)+1]*(frac1) 
                
            #Linea 2
                retardo2 = Nb-muestra + Ret2
                Nretardo2=np.floor(retardo2)      # Redondeo del retardo a nº de muestras enteras.
                frac2=retardo2-Nretardo2           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
                y2=Nbuf[int(Nretardo2)]*(1-frac2)+Nbuf[int(Nretardo2)+1]*(frac2) 
            
            # Solapamos ambas líneas
                yb[muestra]=Gan1*y1+Gan2*y2

        
        
            out_data=1*yb
           
        return (bytes(out_data.astype(np.int16)),flag)
        
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fss,
                input=True,
                output=True,
                start=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)    
    
    def on_button_clickedDet(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True

    def on_button_clickedEmp(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True       
        

        
    botonDet.on_click(on_button_clickedDet)
    botonEmp.on_click(on_button_clickedEmp)

def estima_pitch(tramaanalisis,fs,gEmedia=0,fmin=60,fmax=880,Tol=0.2):
    
    Emedia=np.mean(np.square(np.float64(tramaanalisis)))
    if Emedia>gEmedia:
        alfa=0.1
    else:
        alfa=0.999
    gEmedia=alfa*gEmedia+(1-alfa)*Emedia
                    
    if Emedia==0:
        P=0
        
    else:
        a,Pe= lpc(tramaanalisis,2+int(np.ceil(fs/1000)))
        error=np.float64(signal.lfilter(a,1,tramaanalisis))
        Cx=signal.correlate(error,error)
        
        Npitchmin=int(fs/fmax)
        Npitchmax=int(fs/fmin)
    
        Cxx=Cx[len(error)-1:len(error)-1+Npitchmax]/Cx[len(error)-1]
        Cxx[0:Npitchmin]=0
        Amaxi=np.max(Cxx)  # Máximo de la autocorrelación (para determinar si la trama es sorda o sonora)
        Imaxi=np.argmax(Cxx) # posición del maximo de la autocorrelación
        Cxx[0:Imaxi+Imaxi//2]=0
        Imaxi2=np.argmax(Cxx) # Buscamos si hay un segundo máximo en el entorno del periodo doble
        
        if (Imaxi2/(2*Imaxi)<1+Tol and Imaxi2/(2*Imaxi)>1-Tol) and Emedia>0.05*gEmedia:
            P=fs/Imaxi 
        else:
            P=0
        
    return P,gEmedia    


def ajustaf0(f0,intervalo=1):
    pitch=np.rint(69+(12/intervalo)*np.log2(f0/440))
    f=440*2**((np.rint(pitch)-69)/12*intervalo)
    return f

def autotune_tr(fs,Tb,Tdg=0.03,x=np.array([None,None])):
    '''Realiza en tiempo real el efecto de Autotune.

      Argumentos de entrada:
        fs (escalar): frecuencia de muestreo
        Tb (escalar): Duración del bloque de datos
        Td (escalar): Máximo retardo de la línea de retardo para realizar el Pitch Shifting. Por defecto 30ms 
        x: Señal a representar o NONE para analizar la señal captada por el micrófono'''
    
    
    global In,fss,CHUNK,xss,Nbuf,m,Td,n,numtramas,Nb,ph1,ph2,phD,Overlap,fgain,sd,g_en,P0,Ret1,Ret2,muestra2
    
    Ret1=0
    Ret2=0
    g_en=0
    B=int(np.ceil(fs*Tb))
    Td=1*Tdg
    Overlap=0.1  # Porcentaje de solapamiemto entre las dos líneas de retardo (10%)
    fgain = 1 / Overlap
    Nb=int(np.ceil(fs*Tb))  # Muestras del bloque
    sd=int(np.ceil(fs*Td))  # Muestras del retardo 

    Nbuf=np.zeros(2+Nb+sd)  # Buffer de datos incluyendo líneas de retardo

    CHUNK=1*B 
    
    ph1 = 0                 # fases normalizadas de las funciones de las líneas de retardo
    ph2 = (1 - Overlap)


    In=False
   
    if x[0]==None:
        In=True
    else:
        numtramas=np.ceil(len(x)/CHUNK)
        xss=np.zeros(int(numtramas)*CHUNK)
        xss[0:len(x)]=x

    n=0
                 
    fss=1*fs

    # Botones y controles
    botonDet=widgets.Button(
            description='Detener',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    botonEmp=widgets.Button(
            description='Empezar',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )

    barrasemitonos=widgets.IntSlider(
            style = {'description_width': 'auto'},
            value=1,
            min=1,
            max=12,
            step=1,
            description='Nº de semitonos para el autoajuste',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            layout=widgets.Layout(width='90%', height='60px')
            )

    
    barraFactor=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=-12,
            max=12,
            step=0.5,
            description='Factor de desplazaminto del Pitch',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=widgets.Layout(width='60%', height='60px')
            )

    barraDesviacion=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=-12,
            max=12,
            step=0.01,
            description='   Correccion Pitch',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=widgets.Layout(width='60%', height='40px')
            )
    Check_AjManual=widgets.Checkbox(
        value=False,
        description='',
        disabled=False
    )


    AjusteManual=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=150,
            min=120,
            max=440,
            step=0.5,
            description='Ajuste Manual de Pitch',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=widgets.Layout(width='100%', height='40px')
            )

    fmod=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=1,
            min=0.05,
            max=5,
            step=0.05,
            description='Frecuencia Modulación',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='90%', height='40px')
            )

    Amod=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=0,
            max=0.001,
            step=0.0001,
            description='Amplitud Modulación',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.4f',
            layout=widgets.Layout(width='90%', height='40px')
            )
        # Visores numéricos
    num_f0=widgets.FloatText(value=0.0, description='f_in', layout= widgets.Layout(width='175px'),disabled=True, readout_format='.2f')
    num_fout=widgets.FloatText(value=0.0,description='f_out', layout= widgets.Layout(width='180px'),disabled=True, readout_format='.2f')
    

        # MOSAICO VISUALIZACIÓN
    display(widgets.VBox([widgets.HBox([botonEmp, botonDet]),widgets.HBox([num_f0,barraDesviacion,num_fout]), barrasemitonos, widgets.HBox([Check_AjManual,AjusteManual]),widgets.HBox([fmod,Amod])]),barraFactor)
    
    # Instancia a clase pyaudio
    p = pyaudio.PyAudio()
    P0=100
    muestra2=0
    
    def callback(in_data, frame_count, time_info, status):
        global In,fss,CHUNK,xss,Nbuf,m,Td,n,numtramas,Nb,ph1,ph2,phD,Overlap,fgain,sd,Ret1,Ret2,g_en,P0,muestra2
        flag= pyaudio.paContinue
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        yb=np.zeros(CHUNK)

        if In==False:
            if n==numtramas*CHUNK:
                n=0
            in_data=xss[n:n+CHUNK]
            n=n+CHUNK
        else:
            in_data=audio_data
        
        # Estimamos Pitch
        P,g_en=estima_pitch(in_data,fss,g_en)
        num_f0.value = P
        if P==0:
            P = P0
        else:
            P0 = P
        if Check_AjManual.value:
            f_obj = AjusteManual.value
        else:
            f_obj=ajustaf0(P,barrasemitonos.value)
        
        k=12*np.log2((f_obj/P))
            
        if k==0:
            m = 0.001
        else:
            m = 1-2**((k+barraFactor.value)/12)

        barraDesviacion.value = k+barraFactor.value
        num_fout.value = f_obj 
        
        pRate = m / Td          # Variación del retardo en muestras por segundo
        pstep = pRate / fss      # Paso de las fases (adimensional)
         
        Nbuf=np.concatenate((in_data[-1::-1],Nbuf[:-Nb]))
        if m==0:
            out_data=1*in_data
            
        else:
            # Procesamos muestra a muestra dentro de cada bloque
            Amd=Amod.value
            Fmd=fmod.value/fss
            for muestra in range(Nb):
                muestra2 += 1
                
                if muestra2>1/Fmd:
                    muestra2=0
                ph1 = (ph1 + pstep + Amd*np.sin(2*np.pi*Fmd*muestra2))%1
                ph2 = (ph2 + pstep + Amd*np.sin(2*np.pi*Fmd*muestra2))%1
                
            # línea de retardo 2 se acerca a su fin. Comenzamos solape con linea de retardo 1
                if ((ph1 < Overlap) and (ph2 >= (1 - Overlap))):
                    Ret1 = sd * ph1 
                    Ret2 = sd * ph2 
                    
                    Gan1  = np.cos((1 - (ph1* fgain)) * np.pi/2)
                    Gan2  = np.cos(((ph2 - (1 - Overlap)) * fgain) * np.pi/2)
                    
            # Línea de retardo 1 está activa
                elif ((ph1 > Overlap) and (ph1 < (1 - Overlap))):
                    # El retardo de la línea 2 se matiene fijo mientras la línea 1 está activa
                    ph2 = 0.0
                    Ret1 = sd * ph1 
             
                    Gan1 = 1.0
                    Gan2 = 0.0
                   
            # Linea de retardo 1 se acerca a su fin. Comenzamos solape con linea de retardo 2
                elif ((ph1 >= (1 - Overlap)) and (ph2 < Overlap)):
                    Ret1 = sd * ph1
                    Ret2 = sd * ph2 
                    
                    Gan1 = np.cos(((ph1 - (1 - Overlap)) * fgain) * np.pi/2)
                    Gan2 = np.cos((1 - (ph2* fgain)) * np.pi/2)
                    
            # Linea de retardo 2 está activa
                elif((ph2 > Overlap) and (ph2 < (1 - Overlap))):
                    # El retardo de la línea 1 se matiene fijo mientras la línea 2 está activa
                    ph1 = 0.0    
                    Ret2 = sd * ph2 
                    
                    Gan1 = 0.0
                    Gan2 = 1.0
                    
            
            # Aplicamos las líneas de retardo
                
            #Linea 1
                  
                retardo1 = Nb-muestra + Ret1  
                Nretardo1=np.floor(retardo1)      # Redondeo del retardo a nº de muestras enteras.
                frac1=retardo1-Nretardo1           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
                y1=Nbuf[int(Nretardo1)]*(1-frac1)+Nbuf[int(Nretardo1)+1]*(frac1) 
                
            #Linea 2
                retardo2 = Nb-muestra + Ret2
                Nretardo2=np.floor(retardo2)      # Redondeo del retardo a nº de muestras enteras.
                frac2=retardo2-Nretardo2           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
                y2=Nbuf[int(Nretardo2)]*(1-frac2)+Nbuf[int(Nretardo2)+1]*(frac2) 
            
            # Solapamos ambas líneas
                yb[muestra]=Gan1*y1+Gan2*y2

        
        
            out_data=1*yb+0*in_data
           
        return (bytes(out_data.astype(np.int16)),flag)
        
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fss,
                input=True,
                output=True,
                start=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)    
    
    def on_button_clickedDet(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True

    def on_button_clickedEmp(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True       
        

        
    botonDet.on_click(on_button_clickedDet)
    botonEmp.on_click(on_button_clickedEmp)
    
def moduladorvoz_tr(fs,Tb,Tdg=0.03,x=np.array([None,None])):
    '''Realiza en tiempo real un efecto de modulación de voz basado en desplazamiento del pitch, filtrados y modulaciones.

      Argumentos de entrada:
        fs (escalar): frecuencia de muestreo
        Tb (escalar): Duración del bloque de datos
        Td (escalar): Máximo retardo de la línea de retardo para realizar el Pitch Shifting. Por defecto 30ms 
        x: Señal a representar o NONE para analizar la señal captada por el micrófono'''
    
    
    global fase,beco,zoe,zof,In,fss,CHUNK,xss,Nbuf,m,Td,n,numtramas,Nb,ph1,ph2,phD,Overlap,fgain,sd,P0,Ret1,Ret2,muestra2
    
    
    zof=np.zeros(300)
    beco=np.zeros(int(0.2*fs))
    beco[0]=1
    zoe=np.zeros(len(beco)-1)
    Ret1=0
    Ret2=0
    B=int(np.ceil(fs*Tb))
    Td=1*Tdg
    Overlap=0.1  # Porcentaje de solapamiemto entre las dos líneas de retardo (10%)
    fgain = 1 / Overlap
    Nb=int(np.ceil(fs*Tb))  # Muestras del bloque
    sd=int(np.ceil(fs*Td))  # Muestras del retardo 
    fase=np.arange(Nb)
    Nbuf=np.zeros(2+Nb+sd)  # Buffer de datos incluyendo líneas de retardo

    CHUNK=1*B 
    
    ph1 = 0                 # fases normalizadas de las funciones de las líneas de retardo
    ph2 = (1 - Overlap)


    In=False
   
    if x[0]==None:
        In=True
    else:
        numtramas=np.ceil(len(x)/CHUNK)
        xss=np.zeros(int(numtramas)*CHUNK)
        xss[0:len(x)]=x

    n=0
                 
    fss=1*fs

    # Botones y controles
    botonDet=widgets.Button(
            description='Detener',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )
    botonEmp=widgets.Button(
            description='Empezar',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            )

    Check_Pitch=widgets.Checkbox(
        value=False,
        description='',
        disabled=False
    )
    
    
    barraFactor=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=-12,
            max=12,
            step=0.5,
            description='Factor de desplazaminto del Pitch',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=widgets.Layout(width='60%', height='40px')
            )

    Check_Filtro=widgets.Checkbox(
        value=False,
        description='',
        disabled=False
    )


    barrafiltro=widgets.FloatRangeSlider(
            style = {'description_width': 'auto'},
            value=[10, fs/2-10],
            min=10,
            max=fs/2-10,
            step=0.5,
            description='Filtro Paso Banda',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=widgets.Layout(width='100%', height='40px')
            )

    Check_vibrato=widgets.Checkbox(
        value=False,
        description='',
        disabled=False
    )
    
    fmod=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=1,
            min=0.05,
            max=5,
            step=0.05,
            description='Frecuencia Modulación vibrato',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='90%', height='40px')
            )           
            
    Amod=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=0,
            max=0.001,
            step=0.0001,
            description='Amplitud Modulación vibrato',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.4f',
            layout=widgets.Layout(width='90%', height='40px')
            )

    Check_tremolo=widgets.Checkbox(
        value=False,
        description='',
        disabled=False
    )
    
    fmodt=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=1,
            min=0,
            max=70,
            step=0.5,
            description='Frecuencia Modulación trémolo',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='90%', height='40px')
            )           
            
    Amodt=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=0,
            max=0.99,
            step=0.01,
            description='Amplitud Modulación trémolo',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.4f',
            layout=widgets.Layout(width='90%', height='40px')
            )
    
    Check_eco=widgets.Checkbox(
        value=False,
        description='',
        disabled=False
    )

    retardo_eco=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0,
            min=0,
            max=0.2,
            step=0.01,
            description='Retardo Eco:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
           )

    amplitud_eco=widgets.FloatSlider(
            style = {'description_width': 'auto'},
            value=0.6,
            min=0,
            max=0.7,
            step=0.01,
            description='Amplitud Eco:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            )



    
    # Visores numéricos
    

        # MOSAICO VISUALIZACIÓN
    display(widgets.VBox([widgets.HBox([botonEmp, botonDet]),widgets.HBox([Check_Pitch,barraFactor]),widgets.HBox([Check_vibrato,fmod,Amod]),widgets.HBox([Check_Filtro,barrafiltro]),widgets.HBox([Check_eco,retardo_eco,amplitud_eco]),widgets.HBox([Check_tremolo,fmodt,Amodt])]))
    
    # Instancia a clase pyaudio
    p = pyaudio.PyAudio()
    P0=100
    muestra2=0
    
    def callback(in_data, frame_count, time_info, status):
        global fase,beco,zoe,zof,In,fss,CHUNK,xss,Nbuf,m,Td,n,numtramas,Nb,ph1,ph2,phD,Overlap,fgain,sd,Ret1,Ret2,g_en,P0,muestra2
        flag= pyaudio.paContinue
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        yb=np.zeros(CHUNK)
        m = 0.0001
        
        
        if In==False:
            if n==numtramas*CHUNK:
                n=0
            in_data=xss[n:n+CHUNK]
            n=n+CHUNK
        else:
            in_data=audio_data
        
        out_data=1*in_data
        if Check_Pitch.value:
            m = 1-2**((barraFactor.value)/12)

            if m==0:
                m = 0.0001 
        
        pRate = m / Td          # Variación del retardo en muestras por segundo
        pstep = pRate / fss      # Paso de las fases (adimensional)
         
        Nbuf=np.concatenate((in_data[-1::-1],Nbuf[:-Nb]))
        
        
        if Check_Pitch.value or Check_vibrato.value:
            # Procesamos muestra a muestra dentro de cada bloque
            if Check_vibrato.value:
                Amd=Amod.value
                
            else:
                Amd=0
                
            Fmd=fmod.value/fss
            for muestra in range(Nb):
                muestra2 += 1
                
                if muestra2>1/Fmd:
                    muestra2=0
                ph1 = (ph1 + pstep + Amd*np.sin(2*np.pi*Fmd*muestra2))%1
                ph2 = (ph2 + pstep + Amd*np.sin(2*np.pi*Fmd*muestra2))%1
                
            # línea de retardo 2 se acerca a su fin. Comenzamos solape con linea de retardo 1
                if ((ph1 < Overlap) and (ph2 >= (1 - Overlap))):
                    Ret1 = sd * ph1 
                    Ret2 = sd * ph2 
                    
                    Gan1  = np.cos((1 - (ph1* fgain)) * np.pi/2)
                    Gan2  = np.cos(((ph2 - (1 - Overlap)) * fgain) * np.pi/2)
                    
            # Línea de retardo 1 está activa
                elif ((ph1 > Overlap) and (ph1 < (1 - Overlap))):
                    # El retardo de la línea 2 se matiene fijo mientras la línea 1 está activa
                    ph2 = 0.0
                    Ret1 = sd * ph1 
             
                    Gan1 = 1.0
                    Gan2 = 0.0
                   
            # Linea de retardo 1 se acerca a su fin. Comenzamos solape con linea de retardo 2
                elif ((ph1 >= (1 - Overlap)) and (ph2 < Overlap)):
                    Ret1 = sd * ph1
                    Ret2 = sd * ph2 
                    
                    Gan1 = np.cos(((ph1 - (1 - Overlap)) * fgain) * np.pi/2)
                    Gan2 = np.cos((1 - (ph2* fgain)) * np.pi/2)
                    
            # Linea de retardo 2 está activa
                elif((ph2 > Overlap) and (ph2 < (1 - Overlap))):
                    # El retardo de la línea 1 se matiene fijo mientras la línea 2 está activa
                    ph1 = 0.0    
                    Ret2 = sd * ph2 
                    
                    Gan1 = 0.0
                    Gan2 = 1.0
                    
            
            # Aplicamos las líneas de retardo
                
            #Linea 1
                  
                retardo1 = Nb-muestra + Ret1  
                Nretardo1=np.floor(retardo1)      # Redondeo del retardo a nº de muestras enteras.
                frac1=retardo1-Nretardo1           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
                y1=Nbuf[int(Nretardo1)]*(1-frac1)+Nbuf[int(Nretardo1)+1]*(frac1) 
                
            #Linea 2
                retardo2 = Nb-muestra + Ret2
                Nretardo2=np.floor(retardo2)      # Redondeo del retardo a nº de muestras enteras.
                frac2=retardo2-Nretardo2           # Calculamos el error al redondear para interpolar (en el caso en el que el retardo a aplicar no sea un númeto de muestras entero, interpolamos un valor entre las dos muestras más cercanas).
                y2=Nbuf[int(Nretardo2)]*(1-frac2)+Nbuf[int(Nretardo2)+1]*(frac2) 
            
            # Solapamos ambas líneas
                yb[muestra]=Gan1*y1+Gan2*y2

        
        
            out_data=1*yb+0*in_data
        if Check_Filtro.value:
            b=signal.firwin(301, [barrafiltro.value[0],barrafiltro.value[1]] , fs=fss, pass_zero=False)
            out_data,zof=signal.lfilter(b,1,out_data,zi=zof)
        
        if Check_eco.value:
            beco=np.zeros(int(0.2*fs))
            beco[0]=1
            beco[int(np.rint(fs*retardo_eco.value))]=amplitud_eco.value
            out_data,zoe=signal.lfilter(np.array([1]),beco,out_data,zi=zoe)


        if Check_tremolo.value:    
            
            modulador=Amodt.value*np.cos(2*np.pi*(fmodt.value/fss)*fase)
            fase += CHUNK
            out_data=0.5*(modulador+1)*out_data
        
        return (bytes(out_data.astype(np.int16)),flag)
        
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fss,
                input=True,
                output=True,
                start=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)    
    
    def on_button_clickedDet(b):
        stream.stop_stream()
        stream.close()
        p.terminate()
        botonDet.disabled=True

    def on_button_clickedEmp(b):
        stream.start_stream()
        botonDet.disabled=False
        botonEmp.disabled=True       
        

        
    botonDet.on_click(on_button_clickedDet)
    botonEmp.on_click(on_button_clickedEmp)

def sintesis_aditiva(f0=440,fs=8000,T=1,A=1,Fhi=[None,None], ADSR=[None,None]):
    if ADSR[0]==None:
        n=np.arange(int(np.ceil(fs*T)))
    else:
        n=np.arange(len(ADSR))
    K=np.size(A)
    if Fhi[0]==None:
        Fhi=np.zeros(K)
    
    s=np.zeros(len(n))
    for k in range(K):
        s += np.cos(2*np.pi*(f0/fs)*(k+1)*n)
    
    if ADSR[0]==None:
        s=s/np.max(np.abs(s))
    else:
        s=s*ADSR
        s=s/np.max(np.abs(s))
    
    return s

def sintesis_aditiva_tr(f0=440,Amp=1,Fhi=[None,None],Duracion=1,fs=8000,tmax=0.015,tdecay=0.25,trelease=0.2,Adecay=0.25,Areceso=0.1,fmod=2,Amod=0.25):
    A=np.linspace(0,1,int(tmax*fs))   # Ataque
    D=np.linspace(1,Adecay,int(tdecay*fs)) # Decaimiento
    R=np.linspace(Areceso,0,int(trelease*fs)) # Receso

    Ns=int(Duracion*fs)-len(A)-len(D)-len(R)
    if Ns<=0:
        ADSR=np.concatenate((A,D,R))
    else:
        S=np.linspace(Adecay,Areceso,Ns)
        ADSR=np.concatenate((A,D,S,R))
              
    Amod=(1-Amod)+Amod*np.cos(2*np.pi*np.arange(len(ADSR))*fmod/fs)

    ADSR = ADSR*Amod
    s=sintesis_aditiva(f0=f0,A=Amp,ADSR=ADSR)
    sonido(s,fs)
    plt.plot(np.arange(len(ADSR))/fs,ADSR)
    plt.plot(np.arange(len(ADSR))/fs,s)
    plt.title('curva ADSR y forma de onda')
    plt.xlabel('t(seg)');  

def sintesis_substractiva_tr(señal='MultiTono',f0=440,Duracion=1,fs=8000,ciclo=0.5,Filtro=[100,3000],ADSR=False,tmax=0.015,tdecay=0.25,trelease=0.2,Adecay=0.25,Areceso=0.1,fmod=2,Amod=0.25):
    if ADSR==True:
        A=np.linspace(0,1,int(tmax*fs))   # Ataque
        D=np.linspace(1,Adecay,int(tdecay*fs)) # Decaimiento
        R=np.linspace(Areceso,0,int(trelease*fs)) # Receso

        Ns=int(Duracion*fs)-len(A)-len(D)-len(R)
        if Ns<=0:
            ADSR=np.concatenate((A,D,R))
        else:
            S=np.linspace(Adecay,Areceso,Ns)
            ADSR=np.concatenate((A,D,S,R))
              
        Amod=(1-Amod)+Amod*np.cos(2*np.pi*np.arange(len(ADSR))*fmod/fs)

        ADSR = ADSR*Amod
    else:
        ADSR = np.ones(int(fs*Duracion))
    Duracion=len(ADSR)/fs
    if señal=='MultiTono':
        s=sintesis_aditiva(f0=f0,fs=fs,T=Duracion,A=np.ones(int(np.floor((fs/2)/f0))),ADSR=ADSR)
        s=s-np.mean(s)
    elif señal=='Cuadrada':
        t = np.linspace(0, Duracion, int(Duracion*fs), endpoint=False)
        s = signal.square(2 * np.pi * f0 * t,ciclo)
        s = s-np.mean(s) # Eliminamos la componente de continua si la tiene
        s = s*ADSR 
    
    elif señal=='Triangular':
        t = np.linspace(0, Duracion, int(Duracion*fs), endpoint=False)
        s = signal.sawtooth(2 * np.pi * f0 * t,ciclo)
        s = s-np.mean(s) # Eliminamos la componente de continua si la tiene
        s = s*ADSR 
        
    elif señal=='Ruido':
        s=np.random.randn(len(ADSR))*ADSR
        
    b = signal.firwin(301,Filtro,fs=fs,pass_zero=False)
    s = signal.lfilter(b,1,s)
    

    S, f = espectro(s,fs=fs)
    plt.figure(figsize=(10,3))
    plt.subplot(121),plt.plot(f, S)
    plt.xlabel('Frecuencia (Hz)')
    plt.title('Espectro de señal');
    
    plt.subplot(122),plt.plot(np.arange(len(ADSR))/fs,ADSR)
    plt.plot(np.arange(len(ADSR))/fs,s)
    plt.title('curva ADSR y forma de onda')
    plt.xlabel('t(seg)');  
    sonido(s,fs)
    
def sintesis_fm(fc=440,fs=8000,T=1,Am=0, fm=0, Fhim=0, ADSR=[None,None]):
    if ADSR[0]==None:
        n=np.arange(int(np.ceil(fs*T)))
    else:
        n=np.arange(len(ADSR))
    
    s = np.cos(2*np.pi*(fc/fs)*n + Am*np.cos(2*np.pi*(fm/fs)*n+Fhim))
    
    if ADSR[0]==None:
        s=s/np.max(np.abs(s))
    else:
        s=s*ADSR
        s=s/np.max(np.abs(s))
    
    return s


def sintesis_fm_tr(Duracion=1,fc=440,fs=8000,fm=2,Am=0.25,ADSR=False,tmax=0.015,tdecay=0.25,trelease=0.2,Adecay=0.25,Areceso=0.1):
    if ADSR==True:
        A=np.linspace(0,1,int(tmax*fs))   # Ataque
        D=np.linspace(1,Adecay,int(tdecay*fs)) # Decaimiento
        R=np.linspace(Areceso,0,int(trelease*fs)) # Receso

        Ns=int(Duracion*fs)-len(A)-len(D)-len(R)
        if Ns<=0:
            ADSR=np.concatenate((A,D,R))
        else:
            S=np.linspace(Adecay,Areceso,Ns)
            ADSR=np.concatenate((A,D,S,R))
    else:
        ADSR = np.ones(int(fs*Duracion))
    
    Duracion=len(ADSR)
    n=np.arange(Duracion)
    
    s = np.cos(2*np.pi*(fc/fs)*n + Am*np.cos(2*np.pi*(fm/fs)*n))*ADSR
    

    S, f = espectro(s,fs=fs)
    plt.figure(figsize=(10,3))
    plt.subplot(121),plt.plot(f, S)
    plt.xlabel('Frecuencia (Hz)')
    plt.title('Espectro de señal');
    
    plt.subplot(122),plt.plot(n/fs,ADSR)
    plt.plot(n/fs,s)
    plt.title('curva ADSR y forma de onda')
    plt.xlabel('t(seg)');  
    sonido(s,fs)
