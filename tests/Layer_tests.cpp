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

    int batch_size = 1, batch_limit=20;
    double lr = 6e-3;//If increased above a threshhold the gradients will explode.


    libdl::layers::Convolution2D conv1_1("conv1_1", 3, 16, 1, 1, 1, 28*28); //28x28x1
    libdl::layers::Convolution2D conv1_2("conv1_2", 3, 32, 0, 1, 16, 28*28);//not used
    libdl::layers::ReLU relu1;//28x28x1

    libdl::layers::MaxPool pool1(2, 2);//14x14x16


    libdl::layers::Convolution2D conv2_1("conv2_1", 3, 32, 0, 1, 16, 3*3*32);//14x14x32
    libdl::layers::Convolution2D conv2_2("conv2_2", 3, 64, 0, 1, 32, 3*3*32);//not used
    libdl::layers::ReLU relu2;
    libdl::layers::MaxPool pool2(2, 2);//13x13x32

    libdl::layers::Flatten flatten(batch_size, 11, 11, 32);//7x7*32


    libdl::layers::DenseLayer2D dense1(1152, 400, "dense1", 288); //224x100
    libdl::layers::ReLU relu3;


    libdl::layers::DenseLayer2D dense2(400, 200, "dense2", 700);
    libdl::layers::ReLU relu4;

    libdl::layers::DenseLayer2D dense3(200, 10, "dense3", 350);


    libdl::error::CrossEntropy cross_entropy_error(10);

    libdl::TensorWrapper_Exp batch(batch_size, 28, 28, 1, false);


    libdl::TensorWrapper_Exp out_conv(batch_size, 28, 28, 1, false);
    Eigen::MatrixXd out_dense(batch_size, 4608);

    libdl::TensorWrapper_Exp conv_grads(batch_size, 3, 3, 16, false);
    TensorWrapper grads(batch_limit, 10, 1, 1);

    int iteration = 0;

    TensorWrapper batch_labels(batch_limit, train_labels.get_tensor_height(), train_labels.get_tensor_width(), train_labels.get_tensor_depth());
    batch = get_stratified_batch(train_data, train_labels, batch_labels, batch_limit);
    //normalize
    batch.get_tensor() /= 255;

    //TEST
    //PAdding
    Matrixd mat(4, 4);
    mat << 1, 2, 3, 4,
           1, 2, 3, 4,
           1, 2, 3, 4,
           1, 2, 3, 4;

    std::cout << "Matrix cut: \n" << mat(Eigen::all, {0, 0, 1, 2}) << std::endl;
    

    //TEST

    TensorWrapper b_temp(1, 28, 28, 1);

    std::cout << "\nBefore Training predictions.\n";
    std::cout << "===================================================================\n";

    Matrixd predictions(10, 10);
    int index = 0;

    for (int b = 0; b < train_data.get_batch_size()/batch_size && b < 10; b++) {
        iteration += 1;
        

        b_temp.set_tensor(batch.get_tensor().block(b, 0, 1, 28*28), 28, 28, 1);

        out_conv = conv1_1.forward(b_temp);
        //out_conv = conv1_2.forward(out_conv);
        
        out_conv = relu1.forward(out_conv);
        
        out_conv = pool1.forward(out_conv);

        out_conv = conv2_1.forward(out_conv);
        //out_conv = conv2_2.forward(out_conv);
        out_conv = relu2.forward(out_conv);

        out_conv = pool2.forward(out_conv);

        out_conv = flatten.forward(out_conv);


        out_conv = dense1.forward(out_conv);
        out_conv = relu3.forward(out_conv);

        out_conv = dense2.forward(out_conv);
        out_conv = relu4.forward(out_conv);

        out_conv = dense3.forward(out_conv);

        predictions.row(index) = out_conv.get_tensor();
        index++;

    }
    std::cout << "Done" << std::endl;
    Vectord initial_prediction = cross_entropy_error.predictions(predictions, batch_labels.get_tensor().block(0, 0, 10, 1));



    lr = 5e-2;

    std::cout << "\nTRAINING PHASE.\n";
    std::cout << "===================================================================\n";
    
    for(int epoch = 0; epoch < 80; epoch++) {

        if(epoch % 20 == 0 && epoch != 0){
            lr = 2.3/std::sqrt(epoch) *lr;
            std::cout << "New lr: "<< lr << std::endl;
        }

        for (int b = 0; b < train_data.get_batch_size()/batch_size && b < batch_limit; b++) {
            iteration += 1;
            b_temp.set_tensor(batch.get_tensor().block(b, 0, 1, 28*28), 28, 28, 1);

            out_conv = conv1_1.forward(b_temp);
            
            
            out_conv = relu1.forward(out_conv);
            out_conv = pool1.forward(out_conv);

            out_conv = conv2_1.forward(out_conv);
            
            out_conv = relu2.forward(out_conv);
            out_conv = pool2.forward(out_conv);

            out_conv = flatten.forward(out_conv);


            out_conv = dense1.forward(out_conv);
            out_conv = relu3.forward(out_conv);

            out_conv = dense2.forward(out_conv);
            out_conv = relu4.forward(out_conv);

            out_conv = dense3.forward(out_conv);

            //Backward pass
            grads.get_tensor() = cross_entropy_error.get_gradient(out_conv.get_tensor(), batch_labels.get_tensor().block(b, 0, batch_size, 1), iteration);


            grads = dense3.backward(grads, lr);

            grads = relu4.backward(grads, lr);
            grads = dense2.backward(grads, lr);

            grads = relu3.backward(grads, lr);
            grads = dense1.backward(grads, lr);

            conv_grads = flatten.backward(grads);

            conv_grads = pool2.backward(conv_grads, lr);

            conv_grads = relu2.backward(conv_grads, lr);

            
            //std::cout << "Before conv backward" << std::endl;
            conv_grads = conv2_1.backward(conv_grads, lr);

            conv_grads = pool1.backward(conv_grads, lr);

            conv_grads = relu1.backward(conv_grads, lr);
            
            conv_grads = conv1_1.backward(conv_grads, lr);

        }
    }


    std::cout << "\nTESTING PHASE.\n";
    std::cout << "===================================================================\n";


    index = 0;

    for (int b = 0; b < train_data.get_batch_size()/batch_size && b < 10; b++) {
        iteration += 1;
        

        b_temp.set_tensor(batch.get_tensor().block(b, 0, 1, 28*28), 28, 28, 1);

        out_conv = conv1_1.forward(b_temp);
        //out_conv = conv1_2.forward(out_conv);

        out_conv = relu1.forward(out_conv);
        out_conv = pool1.forward(out_conv);

        out_conv = conv2_1.forward(out_conv);
        //out_conv = conv2_2.forward(out_conv);
        out_conv = relu2.forward(out_conv);

        out_conv = pool2.forward(out_conv);

        out_conv = flatten.forward(out_conv);


        out_conv = dense1.forward(out_conv);
        out_conv = relu3.forward(out_conv);

        out_conv = dense2.forward(out_conv);
        out_conv = relu4.forward(out_conv);

        out_conv = dense3.forward(out_conv);

        predictions.row(index) = out_conv.get_tensor();
        index++;

    }
    Vectord p = cross_entropy_error.predictions(predictions, batch_labels.get_tensor().block(0, 0, 10, 1));


    return 0;
}