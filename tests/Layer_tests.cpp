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
<<<<<<< HEAD
#include "Model.h"

=======
>>>>>>> bb4a4c91a2e0d278c2129e2b75a7e74574f0fa81
#include "TensorWrapper.h"
#include "data_handler.h"
#include "pybind11/pybind11.h"

using namespace Eigen;

using namespace libdl;

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

<<<<<<< HEAD
    /*
    std::cout << "All tests ran!\n";

    model::Model<TensorWrapper_Exp> m;

    m.add(new layers::DenseLayer2D(30, 20, "First"));
    m.add(new layers::DenseLayer2D(20, 40, "Second"));
    m.add(new layers::Convolution2D());*/

    libdl::layers::DenseLayer2D dl2d(2, 2, "Test Layer");
    libdl::layers::DenseLayer2D middle(2, 2, "Middle");
    libdl::layers::DenseLayer2D dl2d_1(2, 1, "Test Layer 2");

    libdl::layers::Sigmoid sig1;
    libdl::layers::Sigmoid sig2;
    libdl::layers::Sigmoid sig3;



    MatrixXd input(4, 2);
    input(0, 0) = 0;
    input(0, 1) = 1;

    input(1, 0) = 1;
    input(1, 1) = 0;

    input(2, 0) = 1;
    input(2, 1) = 1;

    input(3, 0) = 0;
    input(3, 1) = 0;

    TensorWrapper_Exp r_input(4, 2, false);
    r_input.set_tensor(input);

    VectorXd labels(4);
    labels(0) = 1;
    labels(1) = 1;
    labels(2) = 0;
    labels(3) = 0;

    libdl::error::ErrorFunctions e(1, labels);


    TensorWrapper_Exp out1;
    TensorWrapper_Exp out2;
    TensorWrapper_Exp out3;
    TensorWrapper_Exp grads;

    double alpha = 0.1;

    std::cout << dl2d.info() << " "
              << middle.info() << " "
              << dl2d_1.info() << std::endl;
    std::cout << "Inputs\n" << r_input.get_tensor() << std::endl; //You pass it by reference so it changes!

    std::cout << "Error: ";
    for (int i = 0; i < 3000; i++){
        out1 = dl2d.forward(r_input);
        out1 = sig1.forward(out1);
        std::cout << "Out:\n" << out1.get_tensor() << std::endl;


        out2 = middle.forward(out1);
        out2 = sig2.forward(out2);
        std::cout << "Out:\n" << out1.get_tensor() << std::endl;

        out3 = dl2d_1.forward(out2);
        out3 = sig3.forward(out3);
        std::cout << "Out:\n" << out1.get_tensor() << std::endl;

        auto err = e.get_error(labels, out3.get_tensor());

        if(i % 100 == 0){
            std::cout << err << " ";
        }

        grads.set_tensor(e.get_gradient()/4);
        std::cout << "Grads:\n" << grads.get_tensor() << std::endl;


        grads = sig3.backward(grads, alpha);
        grads = dl2d_1.backward(grads, alpha);
        std::cout << "Grads:\n" << grads.get_tensor() << std::endl;

        grads = sig2.backward(grads, alpha);
        grads = middle.backward(grads, alpha);
        std::cout << "Grads:\n" << grads.get_tensor() << std::endl;

        grads = sig1.backward(grads, alpha);
        grads = dl2d.backward(grads, alpha);
        std::cout << "Grads:\n" << grads.get_tensor() << std::endl;

    }

    Eigen::MatrixXd in(1, 2);

    in(0, 0) = 1; in(0, 1) = 0;

    TensorWrapper_Exp t_i(1, 2, false);
    t_i.set_tensor(in);

    TensorWrapper_Exp o1;
    TensorWrapper_Exp o2;
    TensorWrapper_Exp o3;

    o1 = dl2d.forward(t_i);
    o1 = sig1.forward(o1);
    o2 = middle.forward(o1);
    o2 = sig2.forward(o2);
    o3 = dl2d_1.forward(o2);
    o3 = sig3.forward(o3);


    std::cout << "\nOutput: \n" << o3.get_tensor() << std::endl;


    return 0;
}
=======
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
    Eigen::MatrixXd grads(batch_limit, 10);

    int iteration = 0;

    TensorWrapper batch_labels(batch_limit, train_labels.get_tensor_height(), train_labels.get_tensor_width(), train_labels.get_tensor_depth());
    batch = get_stratified_batch(train_data, train_labels, batch_labels, batch_limit);
    //normalize
    batch.get_tensor() /= 255;



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
        
        out_conv.set_tensor(relu1.forward(out_conv.get_tensor()),
                            out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
        
        out_conv = pool1.forward(out_conv);

        out_conv = conv2_1.forward(out_conv);
        //out_conv = conv2_2.forward(out_conv);
        out_conv.set_tensor(relu2.forward(out_conv.get_tensor()),
                            out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());

        out_conv = pool2.forward(out_conv);
>>>>>>> bb4a4c91a2e0d278c2129e2b75a7e74574f0fa81

        out_dense = flatten.forward(out_conv);


        out_dense = dense1.forward(out_dense);
        out_dense = relu3.forward(out_dense);

        out_dense = dense2.forward(out_dense);
        out_dense = relu4.forward(out_dense);

        out_dense = dense3.forward(out_dense);

        predictions.row(index) = out_dense;
        index++;

    }
    std::cout << "Done" << std::endl;
    Vectord initial_prediction = cross_entropy_error.predictions(predictions, batch_labels.get_tensor().block(0, 0, 10, 1));



    lr = 3e-3;

    std::cout << "\nTRAINING PHASE.\n";
    std::cout << "===================================================================\n";
    
    for(int epoch = 0; epoch < 40; epoch++) {

        if(epoch % 10 == 0 && epoch != 0){
            lr = 2.3/std::sqrt(epoch) *lr;
            std::cout << "New lr: "<< lr << std::endl;
        }

        for (int b = 0; b < train_data.get_batch_size()/batch_size && b < batch_limit; b++) {
            iteration += 1;
            b_temp.set_tensor(batch.get_tensor().block(b, 0, 1, 28*28), 28, 28, 1);

            out_conv = conv1_1.forward(b_temp);
            
            
            out_conv.set_tensor(relu1.forward(out_conv.get_tensor()),
                    out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
            out_conv = pool1.forward(out_conv);

            out_conv = conv2_1.forward(out_conv);
            
            out_conv.set_tensor(relu2.forward(out_conv.get_tensor()),
                                out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
            out_conv = pool2.forward(out_conv);

            out_dense = flatten.forward(out_conv);


            out_dense = dense1.forward(out_dense);
            out_dense = relu3.forward(out_dense);

            out_dense = dense2.forward(out_dense);
            out_dense = relu4.forward(out_dense);

            out_dense = dense3.forward(out_dense);

            //Backward pass
            grads = cross_entropy_error.get_gradient(out_dense, batch_labels.get_tensor().block(b, 0, batch_size, 1), iteration);


            grads = dense3.backward(grads, lr);

            grads = relu4.backward(grads, lr);
            grads = dense2.backward(grads, lr);

            grads = relu3.backward(grads, lr);
            grads = dense1.backward(grads, lr);

            conv_grads = flatten.backward(grads);

            conv_grads = pool2.backward(conv_grads, lr);

            conv_grads.set_tensor(relu2.backward(conv_grads.get_tensor(), lr),
                    conv_grads.get_tensor_height(), conv_grads.get_tensor_width(), conv_grads.get_tensor_depth());

            
            //std::cout << "Before conv backward" << std::endl;
            conv_grads = conv2_1.backward(conv_grads, lr);

            conv_grads = pool1.backward(conv_grads, lr);

            conv_grads.set_tensor(relu1.backward(conv_grads.get_tensor(), lr),
                    conv_grads.get_tensor_height(), conv_grads.get_tensor_width(), conv_grads.get_tensor_depth());
            
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

        out_conv.set_tensor(relu1.forward(out_conv.get_tensor()),
                            out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
        out_conv = pool1.forward(out_conv);

        out_conv = conv2_1.forward(out_conv);
        //out_conv = conv2_2.forward(out_conv);
        out_conv.set_tensor(relu2.forward(out_conv.get_tensor()),
                            out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());

        out_conv = pool2.forward(out_conv);

        out_dense = flatten.forward(out_conv);


        out_dense = dense1.forward(out_dense);
        out_dense = relu3.forward(out_dense);

        out_dense = dense2.forward(out_dense);
        out_dense = relu4.forward(out_dense);

        out_dense = dense3.forward(out_dense);

        predictions.row(index) = out_dense;
        index++;

    }
    Vectord p = cross_entropy_error.predictions(predictions, batch_labels.get_tensor().block(0, 0, 10, 1));


    return 0;
}