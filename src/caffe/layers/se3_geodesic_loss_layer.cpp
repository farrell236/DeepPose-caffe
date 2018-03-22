#include "caffe/layers/se3_geodesic_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    
    //std::cout << "---------- TESTING LayerSetUp() START ---------" << std::endl;
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    w_alpha = this->layer_param_.se3_geodesic_loss_param().w_alpha();
    w_beta = this->layer_param_.se3_geodesic_loss_param().w_beta();
    bUseRegularisation = this->layer_param_.se3_geodesic_loss_param().use_regularisation();
    
    /*
    std::cout << "w_alpha: " << w_alpha << std::endl;
    std::cout << "w_beta: " << w_beta << std::endl;
    std::cout << "bUseRegularisation: " << bUseRegularisation << std::endl;
    */
    
    caffe_set(36, (Dtype)0., inner_product_mat_at_identity);
    for (int i=0; i<3; i++) { inner_product_mat_at_identity[i*6+i] = 1. / w_alpha; }
    for (int i=3; i<6; i++) { inner_product_mat_at_identity[i*6+i] = 1. / w_beta; }
    
    //std::cout << "---------- TESTING LayerSetUp() END ---------" << std::endl;

    
}
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
    << "Inputs must have the same dimension.";
    //diff_.ReshapeLike(*bottom[0]);
    
    N = bottom[0]->shape(0);
    
}
    
template <typename Dtype>
Dtype SE3GeodesicLossLayer<Dtype>::clip(const Dtype val, const Dtype lower, const Dtype upper){
        
    //return std::fmax(lower, std::fmin(val, upper));
    return fmax(lower, fmin(val, upper));
    
}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::regularize_transformation(const Dtype* transfo,
                                                            Dtype* regularized_transfo){
    
    this->regularize_rotation(transfo,regularized_transfo);
    caffe_copy(3, transfo+3, regularized_transfo+3);
    
}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::jacobian_translation(const Dtype* point,
                                                       Dtype* jacobian){

    Dtype rot_vec[3] = {0};
    this->regularize_rotation(point,rot_vec);
    
    caffe_set(36, (Dtype)0., jacobian);
    
    Dtype jacobian_rot[9] = {0};
    this->jacobian_rotation(rot_vec, jacobian_rot);
    
    Dtype jacobian_trans[9] = {0};
    this->matrix_from_rotation_vector(rot_vec,jacobian_trans);
    
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            jacobian[i*6+j] = jacobian_rot[i*3+j];
            jacobian[(i+3)*6+(j+3)] = jacobian_trans[i*3+j];
        }
    }
    
    
}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::inverse(const Dtype* transfo,
                                          Dtype* inverse_transfo){
    
    Dtype reg_transfo[6] = {0};
    this->regularize_transformation(transfo,reg_transfo);
    
    Dtype inverse_transfo_[6] = {0};
    
    caffe_cpu_scale(3, (Dtype)-1., reg_transfo, inverse_transfo_);
    
    Dtype rot_mat[9] = {0};
    this->matrix_from_rotation_vector(inverse_transfo_,rot_mat);
    
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 3, 1, 3,
                   (Dtype)-1., rot_mat, transfo+3,
                   (Dtype)0., inverse_transfo_+3);
    
    this->regularize_transformation(inverse_transfo_,inverse_transfo);

    
}
    
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::compose(const Dtype* transfo_1,
                                          const Dtype* transfo_2,
                                          Dtype* prod_transfo){
    
    Dtype rot_mat_1[9] = {0};
    this->matrix_from_rotation_vector(transfo_1,rot_mat_1);
    this->closest_rotation_matrix(rot_mat_1, rot_mat_1);
    
    Dtype rot_mat_2[9] = {0};
    this->matrix_from_rotation_vector(transfo_2,rot_mat_2);
    this->closest_rotation_matrix(rot_mat_2, rot_mat_2);
    
    Dtype prod_rot_mat[9] = {0};
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 3, 3, 3,
                   (Dtype)1., rot_mat_1, rot_mat_2,
                   (Dtype)0., prod_rot_mat);
    
    Dtype prod_transfo_[9] = {0};
    
    this->rotation_vector_from_matrix(prod_rot_mat,prod_transfo_);
    
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 3, 1, 3,
                   (Dtype)1., rot_mat_1, transfo_2+3,
                   (Dtype)0., prod_transfo_+3);
    
    caffe_axpy(3, (Dtype)1., transfo_1+3, prod_transfo_+3);
    
    this->regularize_transformation(prod_transfo_,prod_transfo);
    
    
}
    
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::skew_matrix_from_vector(const Dtype* vec, Dtype* skew_vec){
    
    skew_vec[0] = 0;
    skew_vec[1] = -vec[2];
    skew_vec[2] =  vec[1];
    skew_vec[3] =  vec[2];
    skew_vec[4] = 0;
    skew_vec[5] = -vec[0];
    skew_vec[6] = -vec[1];
    skew_vec[7] =  vec[0];
    skew_vec[8] = 0;
    
}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::vector_from_skew_matrix(const Dtype* skew_mat, Dtype* vec){
    
    vec[0] = skew_mat[7];
    vec[1] = skew_mat[2];
    vec[2] = skew_mat[3];
    
}

template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::closest_rotation_matrix(const Dtype* mat, Dtype* rot_mat){
    
    /* Intel® Math Kernel Library LAPACK Examples: SVD */
    // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_row.c.htm
    
    double tmp[9] = {0};
    std::copy(mat, mat+9, tmp);

    /* Local arrays */
    MKL_INT dim = 3, info;
    double s[3], u[9], vt[9], superb[2];
    info = LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'A', 'A', dim, dim, tmp, dim, s, u, dim, vt, dim, superb );
    
    /* Check for convergence */
    if( info > 0 ) {
        printf( "WARNING: The algorithm computing SVD failed to converge.\n" );
    }
    
    /* tmp = u * vt */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                3, 3, 3, 1, u, 3, vt, 3, 0, tmp, 3);
    
    std::copy(tmp, tmp+9, rot_mat);

    /* Compute LU Decomposition for calculating determinant */
    int ipiv[3] = {0};
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, 3, 3, tmp, 3, ipiv);
    
    /* Check for convergence */
    if( info > 0 ) {
        printf( "WARNING: The algorithm computing LU Decomposition failed to converge.\n" );
    }
    
    Dtype determinant = 1.;
    for (int i=0; i<3; i++) {
        determinant *= tmp[i*3+i] * ((ipiv[i] == i+1)?1:-1);
    }
    
    if (determinant < 0) {
        
        std::cout << "closest_rotation_matrix(): determinant = " << determinant << std::endl;
        
        double mat_diag_s[9] = {1,0,0,0,1,0,0,0,-1};
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    3, 3, 3, 1, u, 3, mat_diag_s, 3, 0, tmp, 3);
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    3, 3, 3, 1, tmp, 3, vt, 3, 0, tmp, 3);
        
        std::copy(tmp, tmp+9, rot_mat);
        
    }

    
}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::rotation_vector_from_matrix(const Dtype* rot_mat,
                                                              Dtype* rot_vec){
    
    Dtype closest_rot_mat[9] = {0};
    this->closest_rotation_matrix(rot_mat,closest_rot_mat);
    
    Dtype trace = closest_rot_mat[0] + closest_rot_mat[4] + closest_rot_mat[8];
    Dtype cos_angle = this->clip( .5 * (trace - 1.) , -1., 1.);
    Dtype angle = acos(cos_angle);
    
    Dtype vec[3] = {    closest_rot_mat[7] - closest_rot_mat[5],
                        closest_rot_mat[2] - closest_rot_mat[6],
                        closest_rot_mat[3] - closest_rot_mat[1], };
    
    if (fabs(angle) < epsilon) {
        //std::cout << "rotation_vector_from_matrix(): fabs(angle) < epsilon" << std::endl;
        Dtype fact = (.5 - (trace - 3.) / 12.);
        caffe_scal(3, fact, vec);
        
    } else if (fabs(angle - M_PI) < epsilon) {
        //std::cout << "rotation_vector_from_matrix(): fabs(angle - M_PI) < epsilon" << std::endl;
        Dtype diag[3] = {closest_rot_mat[0] , closest_rot_mat[4] , closest_rot_mat[8]};
        const int a = std::distance(diag, std::max_element(diag, diag + 3));
        const int b = (a+1)%3;
        const int c = (a+2)%3;
        
        Dtype sq_root = sqrt((closest_rot_mat[a*3+a]
                              - closest_rot_mat[b*3+b]
                              - closest_rot_mat[c*3+c] + 1.));
        
        vec[a] = sq_root / 2.;
        vec[b] = (closest_rot_mat[b*3+a] + closest_rot_mat[a*3+b]) / (2. * sq_root);
        vec[c] = (closest_rot_mat[c*3+a] + closest_rot_mat[a*3+c]) / (2. * sq_root);
        
        Dtype fact = angle / sqrt(caffe_cpu_dot(3, vec, vec));
        caffe_scal(3, fact, vec);
        
    } else {
        //std::cout << "rotation_vector_from_matrix(): else" << std::endl;
        Dtype fact = angle / (2. * sin(angle));
        caffe_scal(3, fact, vec);
        
    }
    
    this->regularize_rotation(vec,rot_vec);
    
}


template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::matrix_from_rotation_vector(const Dtype* rot_vec,
                                                              Dtype* rot_mat){
    
    
    Dtype reg_rot_vec[3] = {0};
    this->regularize_rotation(rot_vec,reg_rot_vec);
    
    Dtype angle = sqrt(caffe_cpu_dot(3, reg_rot_vec, reg_rot_vec));
    
    Dtype skew_rot_vec[9] = {0};
    this->skew_matrix_from_vector(reg_rot_vec,skew_rot_vec);
    
    Dtype coef_1, coef_2;
    
    if (fabs(angle)<epsilon) {
        //std::cout << "matrix_from_rotation_vector(): fabs(angle) < epsilon" << std::endl;
        coef_1 = 1. - (angle * angle) / 6.;
        coef_2 = 1. / 2. - (angle * angle);
    } else {
        //std::cout << "matrix_from_rotation_vector(): else" << std::endl;
        coef_1 = sin(angle) / angle;
        coef_2 = (1. - cos(angle)) / (angle * angle);
    }
    
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 3, 3, 3,
                   coef_2, skew_rot_vec, skew_rot_vec,
                   coef_1, skew_rot_vec);
    
    caffe_copy(9, skew_rot_vec, rot_mat);
    
    rot_mat[0] += 1;
    rot_mat[4] += 1;
    rot_mat[8] += 1;
    
}
    

template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::regularize_rotation(const Dtype* rot_vec,
                                                      Dtype* regularized_rot_vec){
    
    if (bUseRegularisation) {
        
        Dtype angle = sqrt(caffe_cpu_dot(3, rot_vec, rot_vec));
        Dtype k = floor(angle / (2. * M_PI) + .5);
        Dtype fact = (1. - 2. * M_PI * k / angle);
    
        if (angle < epsilon) { fact = 1.; }
    
        caffe_cpu_scale(3, fact, rot_vec, regularized_rot_vec);
        
    } else {
        caffe_copy(3, rot_vec, regularized_rot_vec);
    }

}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::jacobian_rotation(const Dtype* point,
                                                    Dtype* jacobian){
    
    Dtype reg_point[3] = {0};
    this->regularize_rotation(point,reg_point);
    
    Dtype angle = sqrt(caffe_cpu_dot(3, reg_point, reg_point));
    Dtype coef_1, coef_2;
    
    if (fabs(angle) < epsilon){
        //std::cout << "jacobian_rotation(): fabs(angle) < epsilon" << std::endl;
        coef_1 = 1. - (angle * angle) / 12.;
        coef_2 = 1. / 12. + (angle * angle) / 720.;
    } else if (fabs(angle - M_PI) < epsilon){
        //std::cout << "jacobian_rotation(): fabs(angle - M_PI) < epsilon" << std::endl;
        coef_1 = angle * (M_PI - angle) / 4.;
        coef_2 = (1. - coef_1) / (angle * angle);
    } else {
        //std::cout << "jacobian_rotation(): else" << std::endl;
        coef_1 = (angle / 2.) / tan(angle / 2.);
        coef_2 = (1. - coef_1) / (angle * angle);
    }
    
    Dtype outer_reg_point[9] = {0};
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 3, 3, 1,
                   (Dtype)1., reg_point, reg_point,
                   (Dtype)0., outer_reg_point);
    
    this->skew_matrix_from_vector(reg_point,jacobian);
    
    caffe_cpu_axpby(9, coef_2, outer_reg_point, (Dtype)0.5, jacobian);
    
    jacobian[0] += coef_1;
    jacobian[4] += coef_1;
    jacobian[8] += coef_1;
    
    
}
    
    
    
template <typename Dtype>
Dtype SE3GeodesicLossLayer<Dtype>::squared_dist(const Dtype* point_a,
                                                const Dtype* point_b){
    
    Dtype log[6] = {0};
    this->log(point_b,point_a,log);
    return this->squared_norm(log,point_a);
    
}
    
    
template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::log(const Dtype* point,
                                       const Dtype* base_point,
                                       Dtype* log){
    
    Dtype reg_base_point[6] = {0};
    this->regularize_transformation(base_point,reg_base_point);
    
    Dtype reg_point[6] = {0};
    this->regularize_transformation(point,reg_point);

    
    
    Dtype inv_reg_base_point[6] = {0};
    this->inverse(reg_base_point, inv_reg_base_point);
    
    Dtype point_near_id[6] = {0};
    this->compose(inv_reg_base_point,reg_point,point_near_id);
    
    Dtype log_from_id[6] = {0};
    this->left_log_from_identity(point_near_id, log_from_id);
    
    Dtype jacobian[36] = {0};
    this->jacobian_translation(reg_base_point,jacobian);
    
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 6, 1, 6,
                   (Dtype)1., jacobian, log_from_id,
                   (Dtype)0., log);
    
}



template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::left_log_from_identity(const Dtype* point,
                                                         Dtype* log){
    
    Dtype reg_point[6] = {0};
    this->regularize_transformation(point,reg_point);
    
    double inv_inner_product[36] = {0};
    std::copy(inner_product_mat_at_identity, inner_product_mat_at_identity+36, inv_inner_product);
    
    int ipiv[6] = {0};
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, 6, 6, inv_inner_product, 6, ipiv);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, 6, inv_inner_product, 6, ipiv);
    
    Dtype inv_inner_product_[36] = {0};
    std::copy(inv_inner_product, inv_inner_product+36, inv_inner_product_);
    
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 6, 1, 6,
                   (Dtype)1., inv_inner_product_, point,
                   (Dtype)0., log);

}
    
    
template <typename Dtype>
Dtype SE3GeodesicLossLayer<Dtype>::squared_norm(const Dtype* vector,
                                                const Dtype* base_point){
    
    return this->inner_product(vector,vector,base_point);
    
}


template <typename Dtype>
Dtype SE3GeodesicLossLayer<Dtype>::inner_product(const Dtype* tangent_vec_a,
                                                 const Dtype* tangent_vec_b,
                                                 const Dtype* base_point){
    
    Dtype inner_prod_mat[36] = {0};
    this->inner_product_matrix(base_point,inner_prod_mat);
    
    Dtype tmp[6] = {0};
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, 6, 6,
                   (Dtype)1., tangent_vec_a, inner_prod_mat,
                   (Dtype)0., tmp);
    
    return caffe_cpu_dot(6, tangent_vec_b, tmp);

}


template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::inner_product_matrix(const Dtype* base_point,
                                                       Dtype* metric_mat){
    
    Dtype reg_base_point[6] = {0};
    this->regularize_transformation(base_point,reg_base_point);
    
    Dtype jacobian[36] = {0};
    this->jacobian_translation(reg_base_point, jacobian);
    
    double inv_jacobian[36] = {0};
    std::copy(jacobian, jacobian+36, inv_jacobian);
    
    int ipiv[6] = {0};
    
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, 6, 6, inv_jacobian, 6, ipiv);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, 6, inv_jacobian, 6, ipiv);
    
    Dtype inv_jacobian_[36] = {0};
    std::copy(inv_jacobian, inv_jacobian+36, inv_jacobian_);
    
    Dtype tmp[36] = {0};
    
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 6, 6, 6,
                   (Dtype)1., inv_jacobian_, inner_product_mat_at_identity,
                   (Dtype)0., tmp);
    
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 6, 6, 6,
                   (Dtype)1., tmp, inv_jacobian_,
                   (Dtype)0., metric_mat);
    
}
    
    

template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
    
    //std::cout << "---------- TESTING FORWARD_CPU() ---------" << std::endl;
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
    
    //std::cout << "---------- TESTING FORWARD_CPU() END ---------" << std::endl;
    

}

template <typename Dtype>
void SE3GeodesicLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom) {

    //std::cout << "---------- TESTING BACKWARD_CPU() ---------" << std::endl;
    
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
    
    //std::cout << "---------- TESTING BACKWARD_CPU() END ---------" << std::endl;

}

#ifdef CPU_ONLY
STUB_GPU(SE3GeodesicLossLayer);
#endif
    
INSTANTIATE_CLASS(SE3GeodesicLossLayer);
REGISTER_LAYER_CLASS(SE3GeodesicLoss);
    
}  // namespace caffe
