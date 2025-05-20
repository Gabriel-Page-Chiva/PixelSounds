ffmpeg ^
-framerate 25 ^
-i fotogramas/frame_%%04d.png ^
-i "%~1" ^
-c:v libx264 ^
-c:a aac ^
-pix_fmt yuv420p ^
-shortest ^
salida.mp4
