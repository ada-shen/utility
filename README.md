Utility
===

Implementation of our paper: “Verifiability and Predictability: Interpreting Utilities of Network Architectures for Point Cloud Processing” ([arxiv](https://arxiv.org/abs/1911.09053v3)), which has been accepted by CVPR2021.

## Preparation

#### Requirement

- Python 3 (recommend Anaconda3)
- Pytorch 0.4.\*
- Tensorflow 1.14.0
- CMake > 2.8
- CUDA 9.0 + cuDNN 7.6.5
- h5py
- yaml

#### Notice
There are six different models in utility. Among them, RSCNN uses the **pytorch** framework, and other models use the **tensorflow** framework. 
When running different models, you need to enter the corresponding directory to compile as follows.

#### Compile Customized TF Operators

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.14, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

```
    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
```

Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

#### Building RSCNN Kernel

    cd Relation-Shape-CNN
    
    mkdir build && cd build
    
    cmake .. && make

#### Dataset

We focus on 3D shape classification in three different datasets (ModelNet40, ShapeNet, 3dMnist). 
Before experiments, you should download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) 
, [ShapeNet](https://shapenet.cs.stanford.edu/) and [3dMnist](https://www.kaggle.com/daavoo/3d-mnist), and change dataset path in ```.yaml``` files.

Note that to measure the information concentration, we build a small dataset **five_background_small** with background,
which can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1kuVCWMrZk94wInehUyYhVw) (access code: ou5c).


## Usage

Our diagnosis pipeline has three main steps: 1.Train models; 2.Calculate sigma; 3.Diagnose utilities of different architectures.

Here, we give an example to diagnose the utility. Other architecture verification experiment can be designed according to the following example.

First, go to the model directory that you want to do diagnosis.

#### Train models

(1) RSCNN

To train original model and model with *multi scale features*, you need to change the
corresponding items in the ``config_cls.yaml``` and run:
```python
python train_cls.py 
```
To train model with other architectures, you need to change the corresponding items 
in the ``config_dws_cls.yaml``` and run:
```python
python train_dws.py 
```
(2) Other models

You should change dataset path to your correspond dataset path, and run:
```python
python train.py 
```

(3)
We also supply our trained models to [Baidu Cloud](https://pan.baidu.com/s/1MNtCbaOtptsRcG8LfPllgA) (access code: chv1).

#### Calculate sigma

We need to calculate sigma before obtaining utilities. You should run:

``` python
python calculate_sigmaf.py
```

Record the sigmaf value, then, you can evaluate the utility of different architectures.

#### Diagnose utilities of different architectures

#### For neighbourhood consistency:


``` python
python train_layer.py
```



#### For rotation robustness:


``` python
python train_rotate.py
```


#### For adversarial robustness:


``` python
python train_ad.py
```

## Citation

If you use this project in your research, please cite it.

```
@inproceedings{shen2021utility,
 title={Verifiability and Predictability: Interpreting Utilities of Network Architectures for Point Cloud Processing},
 author={Shen, Wen and Wei, Zhihua and Huang, Shikun and Zhang, Binbin and Chen, Panyue and Zhao, Ping and Zhang, Quanshi},
 booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
 year={2021}
}
```
