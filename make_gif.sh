
ffmpeg -i output_euler_distributed_4096single/rho%06d.png output_euler_distributed_4096single.mp4

ffmpeg -y -i output_euler_distributed_4096single.mp4 -vf fps=20,scale=300:-1:flags=lanczos,palettegen palette.png

ffmpeg -i output_euler_distributed_4096single.mp4 -i palette.png -filter_complex "fps=10,scale=300:-1:flags=lanczos[x];[x][1:v]paletteuse"  output_euler_distributed_4096single.gif

rm palette.png