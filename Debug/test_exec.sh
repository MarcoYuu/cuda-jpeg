#!/bin/bash

file_name=(64 128 256 512 768 1024 2056)
for file in ${file_name[@]}; do
	./cuda_jpeg Lena$file.bmp $file.bmp
done
