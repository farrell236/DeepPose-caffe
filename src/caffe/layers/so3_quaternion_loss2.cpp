#include "caffe/layers/so3_quaternion_loss2.hpp"

namespace caffe {

template <typename Dtype>
void SO3QuaternionLoss2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
    CHECK_EQ(bottom[0]->count(1), 4)<< "Quaternions have 4 parameters!.";
    CHECK_EQ(bottom[1]->count(1), 4)<< "Quaternions have 4 parameters!.";
    
    count = bottom[0]->count();
    N = bottom[0]->shape(0);

}

template <typename Dtype>
void SO3QuaternionLoss2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::Reshape(bottom, top);
    
    vector<int> shape_output(1);
    shape_output[0] = bottom[0]->shape(0);
    
    dist_neg.Reshape(shape_output);
    dist_pos.Reshape(shape_output);
    
    vec_neg.ReshapeLike(*bottom[0]);
    vec_add.ReshapeLike(*bottom[0]);
    
}

template <typename Dtype>
void SO3QuaternionLoss2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
    
    Dtype* loss = top[0]->mutable_cpu_data(); loss[0] = 0.;
    
    Dtype* dist_neg_data = dist_neg.mutable_cpu_data();
    Dtype* dist_pos_data = dist_pos.mutable_cpu_data();

    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), vec_neg.mutable_cpu_data());
    caffe_add(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), vec_add.mutable_cpu_data());
    
    // Compute Loss
    for (int i = 0; i < N; i++) {
        
        Dtype* vec_neg_data = vec_neg.mutable_cpu_data() + 4*i;
        Dtype* vec_add_data = vec_add.mutable_cpu_data() + 4*i;

        dist_neg_data[i] = caffe_cpu_dot(4, vec_neg_data, vec_neg_data) / Dtype(8);
        dist_pos_data[i] = caffe_cpu_dot(4, vec_add_data, vec_add_data) / Dtype(8);
        
        loss[0] += fmin(dist_neg_data[i], dist_pos_data[i]) ;
      
    }
    
}

template <typename Dtype>
void SO3QuaternionLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* top_diff = top[0]->cpu_diff();
    
    const Dtype* dist_neg_data = dist_neg.cpu_data();
    const Dtype* dist_pos_data = dist_pos.cpu_data();
    
    Dtype scal = 0.25 * top_diff[0];
    
    for (int i = 0; i < N; i++) {
        
        const Dtype* bottom0 = bottom[0]->cpu_data() + i*4;
        const Dtype* bottom1 = bottom[1]->cpu_data() + i*4;
        
        Dtype sign = (dist_neg_data[i] - dist_pos_data[i]) > 0 ? 1 : -1;
        
        if (propagate_down[0]) {
            Dtype* bottom_diff0 = bottom[0]->mutable_cpu_diff() + 4*i;
            caffe_cpu_scale(4, scal * sign, bottom1, bottom_diff0);
            caffe_axpy(4, scal, bottom0, bottom_diff0);
        }
        if (propagate_down[1]) {
            Dtype* bottom_diff1 = bottom[1]->mutable_cpu_diff() + 4*i;
            caffe_cpu_scale(4, scal * sign, bottom0, bottom_diff1);
            caffe_axpy(4, scal, bottom1, bottom_diff1);
        }
    }
    
}

#ifdef CPU_ONLY
STUB_GPU(SO3QuaternionLoss2Layer);
#endif

INSTANTIATE_CLASS(SO3QuaternionLoss2Layer);
REGISTER_LAYER_CLASS(SO3QuaternionLoss2);

}  // namespace caffe
