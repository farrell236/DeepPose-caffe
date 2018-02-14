#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/se3_geodesic_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SE3GeodesicLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
protected:
    SE3GeodesicLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(64, 6, 1, 1)),
    blob_bottom_label_(new Blob<Dtype>(64, 6, 1, 1)),
    blob_top_loss_(new Blob<Dtype>()) {
        // fill the values
        FillerParameter filler_param;
        
        //filler_param.set_std(100);
        //filler_param.set_mean(10);
        
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);
        filler.Fill(this->blob_bottom_label_);
        blob_bottom_vec_.push_back(blob_bottom_label_);
        blob_top_vec_.push_back(blob_top_loss_);
    }
    virtual ~SE3GeodesicLossLayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_label_;
        delete blob_top_loss_;
    }
    
    void TestForward() {
        // Get the loss without a specified objective weight -- should be
        // equivalent to explicitly specifying a weight of 1.
        LayerParameter layer_param;
        SE3GeodesicLossLayer<Dtype> layer_weight_1(layer_param);
        layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        
        // Get the loss again with a different objective weight; check that it is
        // scaled appropriately.
        const Dtype kLossWeight = 3.7;
        layer_param.add_loss_weight(kLossWeight);
        SE3GeodesicLossLayer<Dtype> layer_weight_2(layer_param);
        layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype kErrorMargin = 1e-5;
        EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
        // Make sure the loss is non-trivial.
        const Dtype kNonTrivialAbsThresh = 1e-1;
        EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
        
        //std::cout << "This is a custom layer test" << std::endl;
        
    }
    
    void TestManual(){
        
        const int N = blob_bottom_vec_[0]->shape(0);
        std::cout << "N: " << N << std::endl;
        
        /*
         
         blob0:
         -2.02058, -0.0173334, -0.556361, -0.80401, -1.64088, -0.485825,
         blob1:
         -0.386787, 2.35808, 0.318384, -1.55862, -0.169628, 0.00206226,
         Loss: -3.46764
         
         */
        std::cout << "-------------------- CUSTOM TEST START --------------------" << std::endl;

        
        Dtype* bottom_data0 = blob_bottom_vec_[0]->mutable_cpu_data();
        Dtype* bottom_data1 = blob_bottom_vec_[1]->mutable_cpu_data();
        
        caffe_set(6*N, (Dtype)0., bottom_data0);
        caffe_set(6*N, (Dtype)0., bottom_data1);
        
        bottom_data0[0] = -59.7634;
        bottom_data0[1] =  83.0158;
        bottom_data0[2] = -159.873;
        bottom_data0[3] = -231.399;
        bottom_data0[4] =  48.9788;
        bottom_data0[5] = -111.723;
        
        bottom_data1[0] = -90.8445;
        bottom_data1[1] = 121.202;
        bottom_data1[2] = 131.176;
        bottom_data1[3] = 73.9502;
        bottom_data1[4] = 112.921;
        bottom_data1[5] = 54.4344;
        
        std::cout << "TEST INIT bottom_data0: " << std::endl;
        for (int i=0; i<N; i++) {
            for (int j=0; j<6; j++) {
                std::cout << bottom_data0[i*6+j] << " ";
            } std::cout << std::endl;
        }
        
        std::cout << "TEST INIT bottom_data1: " << std::endl;
        for (int i=0; i<N; i++) {
            for (int j=0; j<6; j++) {
                std::cout << bottom_data1[i*6+j] << " ";
            } std::cout << std::endl;
        }
        
        LayerParameter layer_param;
        SE3GeodesicLossLayer<Dtype> layer(layer_param);
        
        propagate_down_vec_.push_back(1);
        
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        layer.Backward(this->blob_top_vec_, this->propagate_down_vec_, this->blob_bottom_vec_);
        
        std::cout << "-------------------- CUSTOM TEST END --------------------" << std::endl;
        
    }
    
    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_top_loss_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    vector<bool> propagate_down_vec_;
};

TYPED_TEST_CASE(SE3GeodesicLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SE3GeodesicLossLayerTest, TestForward) {
    this->TestForward();
}
/*
TYPED_TEST(SE3GeodesicLossLayerTest, TestManual) {
    this->TestManual();
}
*/
TYPED_TEST(SE3GeodesicLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    SE3GeodesicLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_,0); // testing only bottom_diff0
}

}  // namespace caffe
