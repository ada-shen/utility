TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# cat $TF_INC
# cat $TF_LIB
#/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

#/bin/bash
/usr/local/cuda-10.1/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
#g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I $TF_LIB -I /usr/local/cuda-10.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I$TF_INC -I /usr/local/cuda-10.1/include -I$TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-10.1/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
