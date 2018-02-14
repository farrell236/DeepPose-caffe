#include "caffe/layers/so3_quaternion_loss3.hpp"

namespace caffe {

template <typename Dtype>
void SO3QuaternionLoss3Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
    CHECK_EQ(bottom[0]->count(1), 4)<< "Quaternions have 4 parameters!.";
    CHECK_EQ(bottom[1]->count(1), 4)<< "Quaternions have 4 parameters!.";
    
    N = bottom[0]->shape(0);
    
}

template <typename Dtype>
void SO3QuaternionLoss3Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::Reshape(bottom, top);
    
    vector<int> shape_output(1);
    shape_output[0] = bottom[0]->shape(0);
    
    dot.Reshape(shape_output);
    
}

template <typename Dtype>
void SO3QuaternionLoss3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
    
    Dtype* loss = top[0]->mutable_cpu_data(); loss[0] = 0;
    Dtype* dot_data = dot.mutable_cpu_data();

    // Compute Loss
    for (int i = 0; i < N; i++) {
        
        const Dtype* bottom_data0 = bottom[0]->cpu_data() + 4*i;
        const Dtype* bottom_data1 = bottom[1]->cpu_data() + 4*i;

        dot_data[i] = caffe_cpu_dot(4, bottom_data0, bottom_data1);
        loss[0] += acos(fabs(dot_data[i]));
      
    }

}

template <typename Dtype>
void SO3QuaternionLoss3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* dot_data = dot.cpu_data();
    
    for (int i = 0; i < N; i++) {
        
        const Dtype* bottom0 = bottom[0]->cpu_data() + i*4;
        const Dtype* bottom1 = bottom[1]->cpu_data() + i*4;

        Dtype coeff = Dtype(-1) * Dtype((dot_data[i] > 0)?1.:-1.) / sqrt(1. - (dot_data[i]*dot_data[i]));
        
        if (propagate_down[0]) {
            Dtype* bottom_diff0 = bottom[0]->mutable_cpu_diff() + 4*i;
            caffe_cpu_scale(4, top_diff[0] * coeff, bottom1, bottom_diff0);
        }
        if (propagate_down[1]) {
            Dtype* bottom_diff1 = bottom[1]->mutable_cpu_diff() + 4*i;
            caffe_cpu_scale(4, top_diff[0] * coeff, bottom0, bottom_diff1);
        }
        
    }
    
}

#ifdef CPU_ONLY
STUB_GPU(Phi3LossLayer);
#endif

INSTANTIATE_CLASS(SO3QuaternionLoss3Layer);
REGISTER_LAYER_CLASS(SO3QuaternionLoss3);

}  // namespace caffe
