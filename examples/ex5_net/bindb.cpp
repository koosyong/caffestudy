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
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

using namespace caffe;
using namespace std;
typedef double Dtype;

int main(int argc, char** argv) {
    if (argc != 7){
        cout<<"option: name nData mean0 str0 mean1 str1"<<endl;
        return 0;
    }
    int dim = 2;
    const char* path = argv[1];
    int nData = atoi(argv[2]);
    Dtype mean_0 = atof(argv[3]);
    Dtype str_0 = atof(argv[4]);
    Dtype mean_1 = atof(argv[5]);
    Dtype str_1 = atof(argv[6]);

    // generate data from two 2-d gaussians
    Blob<Dtype>* const blob_data = new Blob<Dtype>(nData, dim, 1, 1);
    caffe_rng_gaussian<Dtype>(nData/2*dim, mean_0, str_0, blob_data->mutable_cpu_data());
    caffe_rng_gaussian<Dtype>(nData/2*dim, mean_1, str_1, blob_data->mutable_cpu_data()+blob_data->offset(nData/2,0,0,0));

    // open leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    leveldb::WriteBatch* batch = NULL;
    leveldb::Status status = leveldb::DB::Open(options, path, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << path
                       << ". Is it already existing?";
    batch = new leveldb::WriteBatch();

    // save to db
    const int kMaxKeyLength = 10;
    char key_cstr[kMaxKeyLength];

    for (int item_id = 0; item_id < nData; ++item_id) {
        Datum datum;
        datum.set_channels(dim);
        datum.set_height(1);
        datum.set_width(1);

        datum.mutable_float_data()->Reserve(dim);
        for(int k=0;k<2;k++){
            datum.add_float_data(blob_data->cpu_data()[blob_data->offset(item_id,k,0,0)]);
        }

        if(item_id < nData/2)  datum.set_label(0);
        else datum.set_label(1);

        snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
        string keystr(key_cstr);
        batch->Put(keystr, datum.SerializeAsString());
    }

    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    delete db;

    return 0;
}
