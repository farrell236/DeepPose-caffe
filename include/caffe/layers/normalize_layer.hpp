#ifndef CAFFE_NORMALIZE_LAYER_HPP_
#define CAFFE_NORMALIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    
/**
 * @brief Normalizes input.
 
 @Misc{NormalizeLayer,
 author = {Brett Kuprel},
 title={{Normalization (per N) Layer in Caffe}},
 howpublished={URL https://github.com/kuprel/caffe.git},
 year={2015}
 }
 
 Modified to run in latest caffe version
 
 */

template <typename Dtype>
class NormalizeLayer : public Layer<Dtype> {
    
public:
    explicit NormalizeLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
        /*
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
*/
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "NormalizeLayer"; }
    
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    
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
    
};


}  // namespace caffe

#endif  // CAFFE_NORMALIZE_LAYER_HPP_
