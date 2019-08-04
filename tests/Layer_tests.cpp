//
// Created by Aldi Topalli on 2019-05-08.
//

#define CATCH_CONFIG_RUNNER

#include <iostream>
#include "catch.hpp"
#include "Eigen/Dense"
#include "TensorWrapper_tests.cpp"
#include "Layer.h"
#include "ErrorFunctions.h"
#include "TensorWrapper.h"
#include "data_handler.h"
#include "pybind11/pybind11.h"
#include "Model.h"

using namespace Eigen;


TensorWrapper get_stratified_batch(TensorWrapper& data, TensorWrapper& labels, TensorWrapper& batch_labels, int batch_size){
    TensorWrapper batch(20, 28, 28, 1);
    std::vector<int> labels_cnt(10);
    int index = 0;

    for(int img = 0; img < data.get_batch_size(); img++){
        if(labels_cnt[labels.get_tensor()(img)] == 2)
            continue;
        else{
            batch.update_slice(index, 0, data.get_slice(img, 0));
            batch_labels.update_slice(index, 0, labels.get_slice(img, 0));
            labels_cnt[labels.get_tensor()(img)]++;
            index++;
        }

        if(index == batch_size)
            break;
    }

    return batch;
}

int main(int argc, char* argv[]){


    std::cout << "Running tests...\n";
    int result = Catch::Session().run(argc, argv);

    std::cout << "Downloading data.\n";
    std::unique_ptr<data_handler> dh = std::make_unique<data_handler>();

    dh->read_feature_vector("../data/train-images-idx3-ubyte");
    std::cout << "Downloading features.\n";
    dh->read_feature_label("../data/train-labels-idx1-ubyte");
    std::cout << "Downloading labels.\n";
    dh->split_data();
    dh->count_classes();

    libdl::TensorWrapper_Exp train_data   = dh->convert_training_data_to_Eigen();

    libdl::TensorWrapper_Exp train_labels = dh->convert_training_labels_to_Eigen();
    dh.reset(nullptr);

    //Testing maxpool
    TensorWrapper test(2, 4, 4, 3, true);
    libdl::layers::MaxPool pool(2, 2);

    std::cout << "Before pool: \n" << test.get_slice(1, 2) << std::endl;
    test = pool.forward(test);
    std::cout << "After pool:\n" << test.get_slice(1, 2) << std::endl;
    test = pool.backward(test, 0.01);
    std::cout << "After backward: \n" << test.get_slice(1, 2) << std::endl;

    //Testing maxpool
    

    libdl::model::Model model(5, 1e-4, 1, 16, 5, "", "cross_entropy", 10);

    model.add(new libdl::layers::Convolution2D("conv1", 7, 16, 1, 1, 1, 16));
    //model.add(new libdl::layers::MaxPool(2, 2));
    model.add(new libdl::layers::ReLU());
    model.add(new libdl::layers::Convolution2D("conv2", 7, 32, 1, 1, 16, 16));
    //model.add(new libdl::layers::MaxPool(2, 2));
    model.add(new libdl::layers::ReLU());
    model.add(new libdl::layers::DenseLayer2D(12800, 8000, "dense3", 10));
    model.add(new libdl::layers::ReLU());
    model.add(new libdl::layers::DenseLayer2D(8000, 4000, "dense3", 10));
    model.add(new libdl::layers::ReLU());
    model.add(new libdl::layers::DenseLayer2D(4000, 10, "dense4", 10));

    

    //batch.get_tensor() = train_data.get_tensor().block(0, 0, 4, 784);
    //labels.get_tensor() = train_labels.get_tensor().block(0, 0, 4, 1);

    TensorWrapper out(1, 1, 1, 1), grads(1, 1, 1, 1);
    
    int epochs = 3;
    int batch_size = 16;

    TensorWrapper batch(batch_size, 28, 28, 1);
    TensorWrapper labels(batch_size, 1, 1, 1);

    for(int epoch = 0; epoch < epochs; epoch++){
        std::cout << "Epoch " << epoch << std::endl;
        
        for (int b = 0; b < train_data.get_tensor().rows() && b < 100; b++){
            batch.get_tensor() = train_data.get_tensor().block(b, 0, batch_size, 784);
            labels.get_tensor() = train_labels.get_tensor().block(b, 0, batch_size, 1);


            out = model.forward(batch);
            //std::cout << "Out shape: " << out.shape() << std::endl;
            grads = model.backward(out, labels);

        }

        model.set_lr(1 / (1+ 1 * epoch) );
    }


    out = model.forward(batch);



    out.get_tensor().colwise() -= out.get_tensor().rowwise().maxCoeff();

    out.get_tensor() = out.get_tensor().unaryExpr([](double e){return std::exp(e);});

    Vectord sums = out.get_tensor().rowwise().sum();

    for(int i = 0; i < out.get_tensor().rows(); i++){
        out.get_tensor().row(i) /= sums(i);
    }


    int cl;
    double acc = 0;
    for(int o = 0; o < out.get_tensor().rows(); o++){
        std::cout << "Logit: " << out.get_tensor().row(o).maxCoeff(&cl) << " Class: " << cl << " Label: " << labels.get_tensor()(o) << std::endl;
        if(cl == labels.get_tensor()(o))
            acc += 1;
    }

    std::cout << "Test accuracy: " << acc/out.get_tensor().rows() << std::endl;

    
    return 0;
}