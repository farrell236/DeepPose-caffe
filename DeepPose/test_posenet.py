'''
Original File from PoseNet (Kendall et al.) Github repo 
src: https://github.com/alexgkendall/caffe-posenet/blob/master/posenet/scripts/test_posenet.py
'''

import numpy as np
import argparse
import math
import caffe
import scipy.io

import sys
sys.path.append('.../path/to/geomstats/')
from geomstats.special_euclidean_group import SpecialEuclideanGroup
SE3_GROUP = SpecialEuclideanGroup(3)

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

results     = np.zeros((args.iter,2))
geodesic    = np.zeros([args.iter])
_y_pred     = np.zeros([args.iter,7])
_y_true     = np.zeros([args.iter,7])

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

#net = caffe.Net('train_posenet.prototxt','posenet_iter_30000.caffemodel',caffe.TEST)

for i in range(0, args.iter):

    net.forward()

    y_pred = np.squeeze(net.blobs['cls3_fc'].data)
    y_true = np.squeeze(net.blobs['label'].data)

    pose_q = np.squeeze(SE3_GROUP.rotations.quaternion_from_rotation_vector(y_true[0:3]))
    pose_x = y_true[3:6]
    predicted_q = np.squeeze(SE3_GROUP.rotations.quaternion_from_rotation_vector(y_pred[0:3]))
    predicted_x = y_pred[3:6]

    #Compute Individual Sample Error
    q1 = pose_q / np.linalg.norm(pose_q)
    q2 = predicted_q / np.linalg.norm(predicted_q)
    d = abs(np.sum(np.multiply(q1,q2)))
    theta = 2. * np.arccos(d) * 180./math.pi
    error_x = np.linalg.norm(pose_x-predicted_x)

    #Compute Geodesic Distance
    geodesic[i] = np.squeeze(SE3_GROUP.left_canonical_metric.squared_dist(y_pred,y_true))

    _y_pred[i,0:4] = np.copy(predicted_q)
    _y_pred[i,4:7] = np.copy(predicted_x)

    _y_true[i,0:4] = np.copy(pose_q)
    _y_true[i,4:7] = np.copy(pose_x)

    results[i,:] = [error_x,theta]

    print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

median_result = np.median(results,axis=0)
mean_result = np.mean(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'
print 'Mean error ', mean_result[0], 'm  and ', mean_result[1], 'degrees.'
print 'Mean G.D. ' , np.mean(geodesic)

#np.savetxt('results.txt', results, delimiter=' ')

result = {}
result['y_pred'] = _y_pred
result['y_true'] = _y_true
result['geodesic'] = geodesic
result['results'] = results
scipy.io.savemat('output_se3.mat', result)

print 'Success!'


