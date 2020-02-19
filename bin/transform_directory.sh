#!/bin/bash

if [ $# -le 1 ]
then
	echo "Usage: ./transform_directory INPUT_DIRECTORY OUTPUT_DIRECTORY"
	echo "Fourier transform a directory of images"
	echo ""
else
	mkdir -p $2
	declare -i number=0

	for file in $1/*;
	do
		$(dirname "$0")/fourier_transform.out $file $2/$number.png 500 500
		number=$((number+1))
	done
fi