ffmpeg ^
-framerate %~1 ^
-i fotogramas/frame_%%04d.png ^
-c:v libx264 ^
-pix_fmt yuv420p ^
%~2