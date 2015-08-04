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
#include "caffe/net.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;
using namespace std;
typedef double Dtype;

int main(int argc, char** argv) {
    // set net
    string proto =
        "name: 'LogReg_train' "
        "layer { "
        "  name: 'data' "
        "  type: 'Data' "
        "  top: 'data' "
        "  top: 'label' "
        "  data_param { "
        "    source: 'train_leveldb' "
        "    batch_size: 200 "
        "  } "
        "} "
        "layer { "
        "  name: 'ip' "
        "  type: 'InnerProduct' "
        "  bottom: 'data' "
        "  top: 'ip' "
        "  inner_product_param { "
        "    num_output: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' "
        "  bottom: 'ip' "
        "  bottom: 'label' "
        "  top: 'loss' "
        "} ";

    NetParameter param_net;
    google::protobuf::TextFormat::ParseFromString(proto, &param_net);

    SolverParameter param_solver;
    param_solver.set_allocated_net_param(&param_net);
    param_solver.set_base_lr(0.01);
    param_solver.set_max_iter(1000);
    param_solver.set_lr_policy("inv");
    param_solver.set_momentum(0.9);
    param_solver.set_gamma(0.0001);
    param_solver.set_snapshot(1000);
    param_solver.set_display(10);
    param_solver.set_solver_mode(SolverParameter_SolverMode_GPU);
    
    // training
    SGDSolver<Dtype> solver(param_solver);
    solver.Solve();
}
