#include "caffe/layers/se3_geodesic_loss_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {

    //std::cout << "---------- TESTING FORWARD_GPU() ---------" << std::endl;
    /*
    std::cout << "bottom0 shape0: " << bottom[0]->shape(0) << std::endl;
    std::cout << "bottom0 shape1: " << bottom[0]->shape(1) << std::endl;
    
    std::cout << "bottom1 shape0: " << bottom[1]->shape(0) << std::endl;
    std::cout << "bottom1 shape1: " << bottom[1]->shape(1) << std::endl;
    */
    Dtype* top_data = top[0]->mutable_cpu_data();
    top_data[0] = 0;
    
    
    for (int i = 0; i < N; i++) {
        
        //std::cout << "Processing idx[" << i << "] of " << N-1 << std::endl;
        
        const Dtype* bottom_data0 = bottom[0]->cpu_data() + i*6;
        const Dtype* bottom_data1 = bottom[1]->cpu_data() + i*6;
        /*
        std::cout << "blob0: " << std::endl;
        std::cout << "y_pred = np.array([ " ;
        for (int j=0; j<6; j++) {
            std::cout << bottom_data0[j] << " , ";
        } std::cout << " ]) " << std::endl;
        
        std::cout << "blob1: " << std::endl;
        std::cout << "y_true = np.array([ " ;
        for (int j=0; j<6; j++) {
            std::cout << bottom_data1[j] << " , ";
        } std::cout << " ]) " << std::endl;
        */
        Dtype loss = this->squared_dist(bottom_data0,bottom_data1);
        
        //std::cout << "Loss: " << loss << std::endl;
        
        top_data[0] += loss;
        
    }
    
    //std::cout << "---------- TESTING FORWARD_GPU() END ---------" << std::endl;
    
}

template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, 
                                               const vector<Blob<Dtype>*>& bottom) {
    
    //std::cout << "---------- TESTING BACKWARD_GPU() ---------" << std::endl;
    
    const Dtype* top_diff = top[0]->cpu_diff();
    
    //if (propagate_down[0]) { std::cout << "CHECKING GRAD OF BLOB0" << std::endl; }
    //if (propagate_down[1]) { std::cout << "CHECKING GRAD OF BLOB1" << std::endl; }
    
    
    // Compute diff for input blob only
    for (int i = 0; i < N; i++) {
        
        //std::cout << "Processing idx[" << i << "] of " << N-1 << std::endl;
        
        const Dtype* bottom_data0 = bottom[0]->cpu_data() + i*6;
        const Dtype* bottom_data1 = bottom[1]->cpu_data() + i*6;
        Dtype* bottom_diff0 = bottom[0]->mutable_cpu_diff() + i*6;
        /*
        std::cout << "blob0: " << std::endl;
        for (int j=0; j<6; j++) {
            std::cout << bottom_data0[j] << " ";
        } std::cout << std::endl;
        
        std::cout << "blob1: " << std::endl;
        for (int j=0; j<6; j++) {
            std::cout << bottom_data1[j] << " ";
        } std::cout << std::endl;
        */
        
        Dtype tangent_vec[6] = {0};
        this->log(bottom_data1,bottom_data0,tangent_vec);
        
        caffe_scal(6, (Dtype)-2., tangent_vec);
        
        Dtype inner_prod_mat[36] = {0};
        this->inner_product_matrix(bottom_data0,inner_prod_mat);
        
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 6, 1, 6,
                       top_diff[0], inner_prod_mat, tangent_vec,
                       (Dtype)0., bottom_diff0);
        
        //std::cout << "top_diff[0]: " << std::endl << top_diff[0] << std::endl;
        
        caffe_scal(3, w_alpha, bottom_diff0);
        caffe_scal(3, w_beta, bottom_diff0+3);
        /*
        std::cout << "bottom_diff0: " << std::endl;
        for (int j=0; j<6; j++) {
            std::cout << bottom_diff0[j] << " ";
        } std::cout << std::endl;
        */
    }
    
    //std::cout << "---------- TESTING BACKWARD_GPU() END ---------" << std::endl;
    
}

INSTANTIATE_LAYER_GPU_FUNCS(SE3GeodesicLossLayer);

}  // namespace caffe
