#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

using namespace caffe;
using namespace std;
typedef double Dtype;

// parameters
int nTrainData = 200;
int nTestData = 100;
int dim = 2;
int clas = 2;
Dtype mean_0 = -1;
Dtype str_0 = 0.7;
Dtype mean_1 = 1;
Dtype str_1 = 0.7;
int nIter = 1000;

int main(int argc, char** argv) {
    Caffe::set_mode(Caffe::CPU);

    // generate training data from two 2-d gaussians
    Blob<Dtype>* const blob_train_data_ = new Blob<Dtype>(nTrainData, dim, 1, 1);
    Blob<Dtype>* const blob_train_label_ = new Blob<Dtype>(nTrainData, 1, 1, 1);

    caffe_rng_gaussian<Dtype>(nTrainData/2*dim, mean_0, str_0, blob_train_data_->mutable_cpu_data());
    caffe_rng_gaussian<Dtype>(nTrainData/2*dim, mean_1, str_1, blob_train_data_->mutable_cpu_data()+blob_train_data_->offset(nTrainData/2,0,0,0));
    for(int n=0; n<nTrainData; n++){
        if(n < nTrainData/2)    blob_train_label_->mutable_cpu_data()[n] = 0;
        else    blob_train_label_->mutable_cpu_data()[n] = 1;
    }

    // show blob label and data
//    for (int n=0;n<nTrainData;n++){
//        cout<<blob_train_label_->mutable_cpu_data()[blob_train_label_->offset(n,0,0.0)]<<": ";
//        for(int k=0;k<dim;k++){
//            cout<<blob_train_data_->mutable_cpu_data()[blob_train_data_->offset(n,k,0,0)]<<" ";
//        }
//        cout<<endl;
//    }

    // set inner product layer
    vector<Blob<Dtype>*> blob_bottom_ip_vec_;
    vector<Blob<Dtype>*> blob_top_ip_vec_;
    Blob<Dtype>* const blob_top_ip_ = new Blob<Dtype>();

    blob_bottom_ip_vec_.push_back(blob_train_data_);
    blob_top_ip_vec_.push_back(blob_top_ip_);

    LayerParameter layer_ip_param;
    layer_ip_param.mutable_inner_product_param()->set_num_output(clas);
    layer_ip_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
    layer_ip_param.mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");

    InnerProductLayer<Dtype> layer_ip(layer_ip_param);
    layer_ip.SetUp(blob_bottom_ip_vec_, blob_top_ip_vec_);

    // set softmax loss layer
    vector<Blob<Dtype>*> blob_bottom_loss_vec_;
    vector<Blob<Dtype>*> blob_top_loss_vec_;
    Blob<Dtype>* const blob_top_loss_ = new Blob<Dtype>();

    blob_bottom_loss_vec_.push_back(blob_top_ip_);
    blob_bottom_loss_vec_.push_back(blob_train_label_);
    blob_top_loss_vec_.push_back(blob_top_loss_);

    LayerParameter layer_loss_param;
    SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
    layer_loss.SetUp(blob_bottom_loss_vec_, blob_top_loss_vec_);\

    // forward and backward iteration
    for(int n=0;n<nIter;n++){
        // forward
        layer_ip.Forward(blob_bottom_ip_vec_, blob_top_ip_vec_);
        Dtype loss = layer_loss.Forward(blob_bottom_loss_vec_, blob_top_loss_vec_);
        cout<<"Iter "<<n<<" loss "<<loss<<endl;

        // backward
        vector<bool> backpro_vec;
        backpro_vec.push_back(1);
        backpro_vec.push_back(0);
        layer_loss.Backward(blob_top_loss_vec_, backpro_vec, blob_bottom_loss_vec_);
        layer_ip.Backward(blob_top_ip_vec_, backpro_vec, blob_bottom_ip_vec_);

        // update weights of layer_ip
        Dtype rate = 0.1;
        vector<shared_ptr<Blob<Dtype> > > param = layer_ip.blobs();
        caffe_scal(param[0]->count(), rate, param[0]->mutable_cpu_diff());
        param[0]->Update();

        // show weight params and derv of the ip layer
//        for(int i=0;i<param[0]->count();i++){
//            cout<<i<<": "<<param[0]->cpu_data()[i]<<", diff: "<<param[0]->mutable_cpu_diff()[i]<<endl;
//        }
//        cout<<endl;
    }

    // generate test data set
    Blob<Dtype>* const blob_test_data_ = new Blob<Dtype>(nTestData, dim, 1, 1);
    Blob<Dtype>* const blob_test_label_ = new Blob<Dtype>(nTestData, 1, 1, 1);
    caffe_rng_gaussian<Dtype>(nTestData/2*dim, -1, 0.7, blob_test_data_->mutable_cpu_data());
    caffe_rng_gaussian<Dtype>(nTestData/2*dim, 1, 0.7, blob_test_data_->mutable_cpu_data()+blob_test_data_->offset(nTestData/2,0,0,0));
    for(int n=0; n<nTestData; n++){
        if(n < nTestData/2) blob_test_label_->mutable_cpu_data()[n] = 0;
        else    blob_test_label_->mutable_cpu_data()[n] = 1;
    }

    // prediction
    vector<Blob<Dtype>*> blob_bottom_ip_test_vec_;
    vector<Blob<Dtype>*> blob_top_ip_test_vec_;
    Blob<Dtype>* const blob_top_ip_test_ = new Blob<Dtype>();

    blob_bottom_ip_test_vec_.push_back(blob_test_data_);
    blob_top_ip_test_vec_.push_back(blob_top_ip_test_);

    layer_ip.Reshape(blob_bottom_ip_test_vec_, blob_top_ip_test_vec_);
    layer_ip.Forward(blob_bottom_ip_test_vec_, blob_top_ip_test_vec_);

    // evaluation
    int score = 0;
    for (int n=0;n<nTestData;n++){
        int label = blob_test_label_->mutable_cpu_data()[blob_train_label_->offset(n,0,0.0)];
        // argmax evaluate
        Dtype score_0 = blob_top_ip_test_->mutable_cpu_data()[blob_top_ip_test_->offset(n,0,0,0)];
        Dtype score_1 = blob_top_ip_test_->mutable_cpu_data()[blob_top_ip_test_->offset(n,1,0,0)];

        if((score_0 > score_1) && (label == 0)) score ++;
        if((score_0 < score_1) && (label == 1)) score ++;

//        cout<<"label "<<label<<": ";
//        if(score_0 > score_1) cout<<"predict: 0"<<endl;
//        else cout<<"predict: 1"<<endl;
    }
    cout<<"Test score: "<<score<<" out of "<<nTestData<<endl;

    return 0;
}


