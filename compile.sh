cmake ..  -DPY_VERSION=3.7  -DWITH_MKL=OFF  -DWITH_MKLDNN=OFF  -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=ON -DWITH_DISTRIBUTE=OFF -DTENSORRT_ROOT=/path/to/TensorRT-8.2.0.6    -DON_INFER=ON       -DWITH_TESTING=ON      -DCUDA_ARCH_NAME=Auto       -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda-11.3  -DCUDNN_ROOT=/path/to/cudnn -DWITH_CUSPARSELT=ON
