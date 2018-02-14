#ifndef CAFFE_SO3_QUATERNION_LOSS2_LAYER_HPP_
#define CAFFE_SO3_QUATERNION_LOSS2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SO3QuaternionLoss2Layer : public LossLayer<Dtype> {
public:
    explicit SO3QuaternionLoss2Layer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "SO3QuaternionLoss2"; }
    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return true;
    }
protected:


    
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    
    Blob<Dtype> vec_neg; // q1 - q2
    Blob<Dtype> vec_add; // q1 + q2
    
    Blob<Dtype> dist_neg; // ||q1 - q2||
    Blob<Dtype> dist_pos; // ||q1 + q2||
    
    int count;
    int N;
    
};

}  // namespace caffe

#endif  // CAFFE_PHI2_LOSS_LAYER_HPP_
