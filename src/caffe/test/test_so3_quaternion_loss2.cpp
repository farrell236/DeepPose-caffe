#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/so3_quaternion_loss2.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SO3QuaternionLoss2LayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
protected:
    SO3QuaternionLoss2LayerTest()
    : blob_bottom_data_(new Blob<Dtype>(5, 4, 1, 1)),
    blob_bottom_label_(new Blob<Dtype>(5, 4, 1, 1)),
    blob_top_loss_(new Blob<Dtype>()) {
        // fill the values
        FillerParameter filler_param;
        
        //filler_param.set_std(10);
        //filler_param.set_mean(10);
        
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);
        filler.Fill(this->blob_bottom_label_);
        blob_bottom_vec_.push_back(blob_bottom_label_);
        blob_top_vec_.push_back(blob_top_loss_);
        
    }
    virtual ~SO3QuaternionLoss2LayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_label_;
        delete blob_top_loss_;
    }
    
    void TestForward() {
        // Get the loss without a specified objective weight -- should be
        // equivalent to explicitly specifying a weight of 1.
        LayerParameter layer_param;
        SO3QuaternionLoss2Layer<Dtype> layer_weight_1(layer_param);
        layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        
        // Get the loss again with a different objective weight; check that it is
        // scaled appropriately.
        const Dtype kLossWeight = 3.7;
        layer_param.add_loss_weight(kLossWeight);
        SO3QuaternionLoss2Layer<Dtype> layer_weight_2(layer_param);
        layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype kErrorMargin = 1e-5;
        EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
        // Make sure the loss is non-trivial.
        const Dtype kNonTrivialAbsThresh = 1e-1;
        EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
        
    }
    
    void TestBackward() {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        SO3QuaternionLoss2Layer<Dtype> layer(layer_param);
        
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
    }
    
    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_top_loss_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SO3QuaternionLoss2LayerTest, TestDtypesAndDevices);

TYPED_TEST(SO3QuaternionLoss2LayerTest, TestForward) {
    this->TestForward();
}

TYPED_TEST(SO3QuaternionLoss2LayerTest, TestGradient) {
    //this->TestBackward();
    
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    SO3QuaternionLoss2Layer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
    
}

}  // namespace caffe
