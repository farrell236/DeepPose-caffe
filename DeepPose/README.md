# Getting Started

This repository is used for predicting and optimising pose on the manifold space. Here we provide layers to perform rotational estimation for SO3 as well as 6DoF pose estimation in SE3. 

The layers are based on the mathematics provided by DQ Huynh et al \[1\] and Miolane et al. \[2\].

## SE3 Geodesic Loss


### Prototxt API

Layer definition for the ```SE3GeodesicLoss```:

```
layer {
  name: "loss3/loss3"
  type: "SE3GeodesicLoss"
  bottom: "cls3_fc"
  bottom: "label"
  top: "loss3/loss3"
  se3_geodesic_loss_param {
    use_regularisation: true
    w_r1: 1.0 
    w_r2: 1.0 
    w_r3: 1.0 
    w_t1: 1.0 
    w_t2: 1.0 
    w_t3: 1.0 
  }
  loss_weight: 1
}
```

Blobs ```bottom[0]``` and ```bottom[1]``` are not interchangeable. ```bottom[0]``` must be the predicted pose vector and ```bottom[1]``` is the ground truth pose vector. The blobs are of shape ```Nx6``` where N is the batch size.

The loss (rotation vector + translation vector) is represented as an element of the Lie algebra of SE(3), i.e. its tangent space at the identity element of the group, denoted se3. It represents the best linear approximation of SE(3) around its identity element. Since the Lie group SE(3) is 6-dimensional, an element of se3 is a 6D vector.

```w_r1, w_r2, w_r3, w_t1, w_t2, w_t3``` are the individual loss weights for each component in the 6-D loss vector. ```loss_weight``` is the overall loss weight from Caffe, this can be set default to 1. ```use_regularisation``` regularises the input rotation vector so that the rotation is always between ```-pi``` and ```+pi```.

### Generating LMDB Database

The loss function can be used for any pose estimation problem. The provided example is a [GoogLeNet](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt)\[3\] network trained on the [KingsCollege](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)\[4\] dataset.

1. Download the [KingsCollege](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset
2. Download [geomstats](https://github.com/ninamiolane/geomstats)
3. Modify ```create_posenet_lmdb_dataset_se3.py``` (change ln13 and ln17 to the appropriate directories)
4. Run python script
5. Create the mean image file ```meanimage.binaryproto``` using the ```compute_image_mean.bin``` binary compiled as part of the Caffe package.


### Training

1. Modify train_posenet_v1.prototxt, and specify location of your dataset.
2. Modify the ```solver_GoogLeNet.prototxt``` to tweak hyper parameters

	The default hyper parameters are:

	```
	# The train/test net protocol buffer definition
	net: "train_posenet_v1.prototxt"
	
	# test_iter specifies how many forward passes the test should carry out.
	test_iter: 100
	
	# Carry out testing every 500 training iterations.
	test_interval: 500
	
	# The base learning rate, momentum and the weight decay of the network.
	base_lr: 0.0001
	momentum: 0.9
	weight_decay: 0.0005
	
	# The learning rate policy
	lr_policy: "step"
	gamma: 0.9
	stepsize: 20000
	
	# Display every 100 iterations
	display: 100
	
	# The maximum number of iterations
	max_iter: 200000
	
	# snapshot intermediate results
	snapshot: 50000
	snapshot_prefix: "posenet"
	
	# Run network on CPU or GPU
	type: "Adam"
	solver_mode: GPU
	
	```

3. To train, call from shell:

```
caffe.bin train --solver=solver_GoogLeNet.prototxt
```

### Evaluating

1. Modify ln13 in ```test_posenet.py```, and specift the appropriate path for geomstats.
2. Run from command line: 

	```
	python test_posenet.py --model train_posenet_v1.prototxt --weights posenet_iter_200000.caffemodel --iter 343 
	```

This will save predicted pose and ground truth pose, as well as the geodesic distance, in a ```.mat``` file.

## SO3 Layers

These layers are for regressing only rotations on the manifold. 

### Prototxt API

```
layer {
  name: "loss3/loss3"
  type: "SO3QuaternionLoss2"
  bottom: "cls3_fc"
  bottom: "label"
  top: "loss3/loss3"
  loss_weight: 1
}
```
The losses for SO3 implements the equation from \[1\]

* ```SO3QuaternionLoss2``` ==> Phi_2 
* ```SO3QuaternionLoss3``` ==> Phi_3 
* ```SO3QuaternionLoss4``` ==> Phi_4  



## Instance Normalization Layer

This layer performs normalisation separately for each example in the batch (i.e. performs the  normalisation operation across axis=0). 

### Prototxt API

```
layer {
  name: "ins_norm0"
  type: "NormalizeLayer"
  bottom: "ins_norm0_in"
  top: "ins_norm0_out"
}
```

This is a modified version of [Brett Kuprel](https://github.com/kuprel/caffe)'s \[5\] custom caffe and updated to run on Caffe v1.0+. 

---

#### References:

\[1\] Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of Mathematical Imaging and Vision 35.2 (2009): 155-164.

\[2\] Miolane, N.: Geomstats: Computations and statistics on manifolds with geometricstructures. (Feb 2018), https://github.com/ninamiolane/geomstats

\[3\] Szegedy, Christian, et al. "Going deeper with convolutions." Cvpr, 2015.

\[4\] Kendall, A., et al.: Posenet: A convolutional network for real-time 6-DOF camerarelocalization. In: ICCV. pp. 2938–2946 (2015)

\[5\] Kuprel, B.: normalize-layer (Feb 2018), https://github.com/kuprel/caffe


