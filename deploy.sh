#!/bin/bash
# This script adds the LibTorch library, compiles and deploys the artemis application on the server.
libtorch_CPU='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip';
libtorch_CUDA='https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip';
libtorch_path='src/'

#STEP 1 - Installing build environment for artemis (object detection and tracking) ...
##sudo add-apt-repository -y ppa:maarten-fonville/protobuf
##sudo apt-get -q update
#sudo apt-get install -y libprotobuf-dev protobuf-compiler libopencv-dev libopencv-imgproc-dev libopencv-highgui-dev libeigen3-dev libgoogle-glog-dev libglfw3-dev libglew-dev libboost-system-dev
#sudo apt-get install -y libasio-dev libssl-dev 

#STEP 2 - Installing CUDA
#Where ${OS} is ubuntu1804 or ubuntu2004. 
#wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 
#sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
#sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
#sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
#sudo apt-get update

#STEP 3 - Installing cuDNN for Ubuntu 18.04 and 20.04
#sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
#sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
#${cudnn_version} is 8.2.4.*
#${cuda_version} is cuda10.2 or cuda11.4

#STEP 3 - Сreating a folder "data"
#mkdir data


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
		arch=$(basename $libtorch_CUDA)
	;;
	*)
		:
	;;
	esac

arch=${arch//'%2B'/'+'}
mkdir $libtorch_path
unzip /tmp/$arch -d $libtorch_path
mkdir build
cd build && cmake .. && cmake --build . --config Release -j 4
# Manual execution of cmake:
# cmake -DCMAKE_CUDA_ARCHITECTURES=all -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DOpenCV_DIR=/opt/xdk/opencv/build -DTORCH_INSTALL_PREFIX=/opt/xdk/libtorch-cxx11-gpu -DTorch_DIR=/opt/xdk/libtorch-cxx11-gpu/share/cmake/Torch ..
