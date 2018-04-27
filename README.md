# DeepPose-caffe

A general Riemannian formulation of the pose estimation problem to train CNNs directly on SO(3) and SE(3) equipped with a left-invariant Riemannian metric. 


## Build and Installation

This package requires building Caffe with Intel MKL.

### Modified Files

* cmake/Summary.cmake
* cmake/Dependencies.cmake
* cmake/Modules/FindMKL.cmake
* include/caffe/layers/base_data_layer.hpp
* src/caffe/layers/data_layer.cpp
* src/caffe/layers/dropout_layer.cpp
* src/caffe/layers/dropout_layer.cu
* src/caffe/proto/caffe.proto


### Added Files

* include/caffe/layers/normalize_layer.hpp
* include/caffe/layers/se3_geodesic_loss_layer.hpp 
* include/caffe/layers/so3_quaternion_loss2.hpp 
* include/caffe/layers/so3_quaternion_loss3.hpp 
* include/caffe/layers/so3_quaternion_loss4.hpp 
* src/caffe/layers/normalize_layer.cpp
* src/caffe/layers/normalize_layer.cu
* src/caffe/layers/se3_geodesic_loss_layer.cpp
* src/caffe/layers/se3_geodesic_loss_layer.cu
* src/caffe/layers/so3_quaternion_loss2.cpp
* src/caffe/layers/so3_quaternion_loss2.cu
* src/caffe/layers/so3_quaternion_loss3.cpp
* src/caffe/layers/so3_quaternion_loss3.cu
* src/caffe/layers/so3_quaternion_loss4.cpp
* src/caffe/layers/so3_quaternion_loss4.cu
* src/caffe/test/test_normalize_layer.cu
* src/caffe/test/test_se3_geodesic_loss_layer.cpp
* src/caffe/test/test_so3_quaternion_loss2.cpp
* src/caffe/test/test_so3_quaternion_loss3.cpp
* src/caffe/test/test_so3_quaternion_loss4.cpp


## Added Layers

These loss functions optimises on the manifold

* SE3 Geodesic Loss (Rotation + Translation)
* SO3 Quaternion Loss (Rotations only)
* Instance Normalisation Layer

## Usage

See DeepPose/README.md

## Authors & Citation

* Benjamin Hou
* Nina Miolane
* Bishesh Khanal
* Bernhard Kainz

If you like our work and found it useful for your research, please cite our paper. Thanks! :)

```
@inproceedings{hou2018computing,
  title={Computing CNN Loss and Gradients for Pose Estimation with Riemannian Geometry},
  author={Hou, Benjamin and Miolane, Nina and Khanal, Bishesh and Lee, Matthew and Alansary, Amir and McDonagh, Steven and Hajnal, Jo V and Rueckert, Daniel and Glocker, Ben and Kainz, Bernhard},
  booktitle={ International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2018},
  organization={Springer}
}
```

```
@misc{miolane2018geomstats, 
  title={Geomstats: Computations and Statistics on Manifolds with Geometric Structures.}, 
  url={https://github.com/ninamiolane/geomstats}, 
  journal={GitHub}, 
  author={Miolane, Nina and Mathe, Johan and Pennec, Xavier}, 
  year={2018}, 
  month={Feb}
}
```

## Acknowledgements

* Miolane et al. [Geomstats](https://github.com/ninamiolane/geomstats)
* Kendall et al. - [Kings College Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)
* Du Q. Hyunh et al. - [Metrics for 3D Rotations: Comparison and Analysis](https://link.springer.com/article/10.1007%2Fs10851-009-0161-2)

---

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
