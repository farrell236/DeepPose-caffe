'''
Original File from PoseNet (Kendall et al.) Github repo 
src: https://github.com/alexgkendall/caffe-posenet/blob/master/posenet/scripts/create_posenet_lmdb_dataset.py
'''

import numpy as np
import lmdb
import caffe
import random
import cv2

import sys
sys.path.append('.../path/to/geomstats/')
from geomstats.special_euclidean_group import SpecialEuclideanGroup
SE3_GROUP = SpecialEuclideanGroup(3)

directory = '.../path/to/KingsCollege/'
dataset = 'dataset_train.txt'

poses = []
images = []

with open(directory+dataset) as f:
    next(f)  # skip the 3 header lines
    next(f)
    next(f)
    for line in f:
        fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        p3 = float(p3)
        p4 = float(p4)
        p5 = float(p5)
        p6 = float(p6)

        q = np.array([p3,p4,p5,p6])
        v = SE3_GROUP.rotations.rotation_vector_from_quaternion(q)[0]

        print (v[0],v[1],v[2],p0,p1,p2)

        poses.append((v[0],v[1],v[2],p0,p1,p2))
        images.append(directory+fname)

r = list(range(len(images)))
random.shuffle(r)

print 'Creating PoseNet Dataset.'
env = lmdb.open('posenet_dataset_train_lmdb', map_size=int(1e12))

count = 0

for i in r:
    if (count+1) % 100 == 0:
        print 'Saving image: ', count+1
    X = cv2.imread(images[i])
    X = cv2.resize(X, (455,256))    # to reproduce PoseNet results, please resize the images so that the shortest side is 256 pixels
    X = np.transpose(X,(2,0,1))
    im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
    im_dat.float_data.extend(poses[i])
    str_id = '{:0>10d}'.format(count)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())
    count = count+1

env.close()

