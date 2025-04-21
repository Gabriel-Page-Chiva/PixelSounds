# Proyecto PixelSounds

## 1. Objetivo
La misión de este proyecto es crear un algoritmo capaz de crear, a partir de una imagen dada, un archivo de audio generado a través de cierto análisis a esta.

Existen muchos parametros y operaciones con las que podemos extraer información que defina a una imagen; la solución más inmediata al problema es entonces usar estas como generadoras de las características de un posible sonido como resultado.

### De momento, ¿qué extraer de las entradas?
- Las distintas representaciones de la imagen (RGB, YcBcR) y canales de estas
- Componentes frecuenciale (DFT), aunque sería interesante otros tipos de transformada (DCT por ejemplo) con las que extraer otras componentes
- Gradiente (derivada o diferencias entre los valores de los píxeles) tanto vertical como horizontal
- Selección de valores concretos de los píxeles (colores o valores de brillo específicos)
- Operaciones geométricas a la imagen
- Histograma

## Sobre el histograma de una imagen

El histograma es un gráfico que nos ayuda a ver cómo están repartidos los niveles de brillo en los canales de una imagen. De normal puede ser útil para ver el nivel de contraste o si está sub- o sobre- expuesta. Sin embargo, ¿cómo podríamos aprovecharlo para nuestro objetivo? 

Una idea podía ser crear señales de audio a partir de este, ya que algunos histogramas de imágenes tienen parecido con algunos de audio, aunque salvando ciertas diferencias. Se pueden ver picos máximos parecidos a los que aparecen sobre el nivel 0 en audio y algunas distribuciones podrían parecerse a la geométrica, sin embargo esta es una operación no invertible, y aunque nos pueda dar información sobre los niveles que podría tener el sonido, no sabemos cómo distribuir estos.

No habría que abandonar al 100% este camino, pero habría que darle una vuelta para poder aprovecharlo.

Otra posibilidad es, y a esta le veo más futuro, poder asociar el uso de efectos a ciertas partes o toda la señal sonora en base al histograma de alguno o todos sus canales de color. Por ejemplo, si se dividiera el histograma del brillo de la imagen en ciertos rangos y donde cayera cierto porcentaje de píxeles en este aplicar de una forma u otra ciertos efectos.

Ejemplo: analizando el histograma de brillo de un bloque de una imagen dada se ha visto que el 60% de todos los píxeles tienen un valor entre 0 y 50 de brillo. Esto provocará que a ese bloque de la imagen se le aplique un efecto de eco múltiple con 3 repeticiones, retardadas 0.3, 0.5 y 0.7 segundos y amplitudes indirectas 0.5, 0.3, y 0.22