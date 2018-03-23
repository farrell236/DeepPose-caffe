#ifndef CAFFE_SE3_GEODESIC_LOSS_LAYER_HPP_
#define CAFFE_SE3_GEODESIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/math_functions.hpp"

#include "caffe/layers/loss_layer.hpp"

#include <mkl.h>


namespace caffe {

template <typename Dtype>
class SE3GeodesicLossLayer : public LossLayer<Dtype> {
public:
    explicit SE3GeodesicLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param), epsilon(1e-12) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "SE3GeodesicLoss"; }
    /**
     * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
     * to both inputs -- override to return true and always allow force_backward.
     */
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
    
    
    
private:
    
    // Generic Functions
    Dtype clip(const Dtype val, const Dtype lower, const Dtype upper);
    

    
    // SE3 Functions
    void regularize_transformation(const Dtype* transfo, Dtype* regularized_transfo);
    void jacobian_translation(const Dtype* point, Dtype* jacobian);
    void inverse(const Dtype* transfo, Dtype* inverse_transfo);
    void compose(const Dtype* transfo_1, const Dtype* transfo_2, Dtype* prod_transfo);
    
    // SO3 Functions
    void skew_matrix_from_vector(const Dtype* vec, Dtype* skew_vec);
    void vector_from_skew_matrix(const Dtype* skew_mat, Dtype* vec);
    void closest_rotation_matrix(const Dtype* mat, Dtype* rot_mat);
    void rotation_vector_from_matrix(const Dtype* rot_mat, Dtype* rot_vec);
    void matrix_from_rotation_vector(const Dtype* rot_vec, Dtype* rot_mat);
    
    void regularize_rotation(const Dtype* rot_vec, Dtype* regularized_rot_vec);
    void jacobian_rotation(const Dtype* point, Dtype* jacobian);
    
    // Riemannian Metric Functions
    Dtype squared_dist(const Dtype* point_a, const Dtype* point_b);
    void log(const Dtype* point, const Dtype* base_point, Dtype* log);
    void left_log_from_identity(const Dtype* point, Dtype* log);
    
    Dtype squared_norm(const Dtype* vector, const Dtype* base_point);
    Dtype inner_product(const Dtype* tangent_vec_a, const Dtype* tangent_vec_b, const Dtype* base_point);
    void inner_product_matrix(const Dtype* base_point, Dtype* metric_mat);
    
    // Constants
    int N;
    
    Dtype w_r1;
    Dtype w_r2;
    Dtype w_r3;
    Dtype w_t1;
    Dtype w_t2;
    Dtype w_t3;

    bool bUseRegularisation;
    
    Dtype inner_product_mat_at_identity[36];
    
    const double epsilon;

    
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
