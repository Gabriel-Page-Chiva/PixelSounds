# Proyecto PixelSounds

**PixelSounds** es el resultado al que hemos llegado entre todos los participantes de este grupo al querer responder, a nuestra manera, la misma cuestión:  

_¿es posible una compatibilidad entre las imágenes y el audio?_

Los métodos y resultados que hemos desarrollado no tienen por qué ser ni los únicos ni los correctos; son una de las posibles interpretaciones que se pueden escoger.

El proceso ha sido un conjunto de probar cosas nuevas, añadir algo para ver qué podría pasar, y observar la evolución de una idea mientras seguíamos nuestra curiosidad por desarrollar conocimientos.

## #Álvaro

Se experimentado con procesos de sintetización de audio a partir de imágenes digitales. Además se ha trabajado en implementar métodos para que ciertos descriptores de las imágenes afecten y modifiquen el contenido o naturaleza de los audios generados.

Realizando una DFT por bloques a lo largo de la imagen se ha creado un conjunto de espectros que se pueden interpretar como una STFT para generar audio MIDI. Se ha probado también realizando DCT's de igual forma.

El audio resultante está compuesto por una serie de tonos de frecuencia adaptada al rango frecuencial de sensibilidad auditiva humano para que no sean demasiado desagradbles al escucharlos.

Cuando se ha pretendido mapear la información de la imagen original en el audio se han tomado:
- Los canales de color para afectar a la duración, amplitud y desviación de frecuencia de los tonos MIDI.
- El histograma de un canal de color de la imagen para afectar a un efecto de vibrato sobre el audio.
- El módulo del gradiente de una imagen para crear coeficientes de un filtro FIR que aplicar al audio generado.
- La entropía de Shannon de una imagen de LBP para recuantificar la señal.

## #RICARDO

Se ha desarrollado un sistema de codificación inversa que transforma señales de audio en una secuencia de imágenes, y permite su posterior reconstrucción para recuperar la señal original.

Este sistema, denominado **PixelSounds**, analiza el audio en bloques temporales y codifica sus características en imágenes (en escala de grises o RGB) que se empaquetan como vídeo. Cada fotograma representa un bloque de audio, y existen distintos modos de codificación:

- **AMPL:** codifica directamente la forma de onda normalizada.
- **FFT:** aplica una Transformada Rápida de Fourier (FFT) por bloque, codificando su magnitud y fase.
- **FIR:** aplica tres filtros FIR (bajo, banda y alto) y codifica la energía de cada banda como un canal RGB (en color) o como filas intercaladas (en gris).

El vídeo resultante contiene toda la información necesaria para reconstruir el audio. Un decodificador extrae los frames, interpreta los datos según el modo y realiza un *overlap-add* con ventana para recomponer la señal.

Este enfoque permite visualizar el contenido de una señal sonora en el dominio de la imagen, abrir vías de codificación reversible o incluso fusionar audio e imagen en un medio común.