ffmpeg ^
-framerate %~1 ^
-i fotogramas/frame_%%04d.png ^
-c:v libx264 ^
-crf 0 ^
-preset veryslow ^
-pix_fmt yuv444p ^
%~2
