#!/bin/bash
# This script adds the LibTorch library, compiles and deploys the artemis application on the server.
libtorch_CPU='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip';
libtorch_CUDA='https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip';
libtorch_path='src/'

while [ -z "${lcp}" ]
do
	echo
	echo "Libtorch Compute Platform"
	echo "1) CPU"
	echo "2) CUDA 11.1"
	echo "3) exit"
	read doing
	case $doing in
	1)
		lcp=1
	;;
	2)
		lcp=2
	;;
	3)
	exit 0
	;;
	*)
		:
	;;
	esac
done

arch='';

case $lcp in
	1)
		wget -P /tmp/ $libtorch_CPU
        arch=$(basename $libtorch_CPU)
	;;
	2)
		wget -P /tmp/ $libtorch_CUDA
        arch=$(basename $libtorch_CPU)
	;;
	*)
		:
	;;
	esac

arch=${arch//'%2B'/'+'}
mkdir $libtorch_path
unzip /tmp/$arch -d $libtorch_path
mkdir build
cd build
cmake ..
cmake --build ./
