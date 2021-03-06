#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {

    top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    int n = bottom[0]->num();
    int d = bottom[0]->count() / n;
    
    for (int i=0; i<n; ++i) {
        Dtype normsqr = caffe_cpu_dot<Dtype>(d, bottom_data+i*d, bottom_data+i*d);
        caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);
    }
    
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    int n = top[0]->num();
    int d = top[0]->count() / n;
    
    for (int i=0; i<n; ++i) {
        Dtype a = caffe_cpu_dot(d, top_data+i*d, top_diff+i*d);
        caffe_cpu_scale(d, a, top_data+i*d, bottom_diff+i*d);
        caffe_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
        a = caffe_cpu_dot(d, bottom_data+i*d, bottom_data+i*d);
        caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
    }
    
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif
    
INSTANTIATE_CLASS(NormalizeLayer);

    
}  // namespace caffe
