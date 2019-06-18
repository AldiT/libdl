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

using namespace Eigen;





int main(int argc, char* argv[]){


    std::cout << "Running tests...\n";
    int result = Catch::Session().run(argc, argv);

    std::unique_ptr<data_handler> dh = std::make_unique<data_handler>();

    dh->read_feature_vector("../data/train-images-idx3-ubyte");
    dh->read_feature_label("../data/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    libdl::TensorWrapper_Exp train_data   = dh->convert_training_data_to_Eigen();
    libdl::TensorWrapper_Exp train_labels = dh->convert_training_labels_to_Eigen();

    libdl::layers::Convolution2D conv1(3, 16, 1, 1, 1); //28x28x1
    libdl::layers::ReLU relu1;//28x28x1

    libdl::layers::MaxPool pool1(3, 2);//14x14x16

    libdl::layers::Convolution2D conv2(3, 32, 1, 1, 1);//14x14x32
    libdl::layers::ReLU relu2;
    libdl::layers::MaxPool pool2(2, 2);//7x7x32

    libdl::layers::Flatten flatten(7, 7, 7, 7);//7x7*32

    libdl::layers::DenseLayer2D dense1(1152, 500, "dense1"); //224x100
    libdl::layers::ReLU relu3;

    libdl::layers::DenseLayer2D dense2(500, 10, "dense2");

    libdl::error::CrossEntropy cross_entropy_error(10,
            train_labels.get_tensor().block(0, 0, 16, 1));


    libdl::TensorWrapper_Exp batch(16, 28, 28, 1, false);
    std::cout << "Batch created.\n";


    libdl::TensorWrapper_Exp out_conv(16, 28, 28, 1, false);
    Eigen::MatrixXd out_dense(16, 500);

    libdl::TensorWrapper_Exp conv_grads(16, 7, 7, 32, false);
    Eigen::MatrixXd grads(16, 10);

    double lr = 0.3;

    std::cout << "Error: ";
    for(int epoch = 0; epoch < 3; epoch++) {
        for (int b = 0; b < train_data.get_batch_size()/16; b++) {
            batch.set_tensor(train_data.get_tensor().block(b, 0, 16, 28*28), 28, 28, 1);
            std::cout << "Batch created.\n";

            out_conv = conv1.forward(batch);
            out_conv.set_tensor(relu1.forward(out_conv.get_tensor()),
                    out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
            out_conv = pool1.forward(out_conv);
            std::cout << "First block.\n";

            out_conv = conv2.forward(out_conv);
            out_conv.set_tensor(relu2.forward(out_conv.get_tensor()),
                                out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
            out_conv = pool2.forward(out_conv);
            std::cout << "Second block.\n";

            out_dense = flatten.forward(out_conv);
            std::cout << "Flatten.\n";

            out_dense = dense1.forward(out_dense);
            out_dense = relu3.forward(out_dense);
            std::cout << "Third block.\n";

            out_dense = dense2.forward(out_dense);
            std::cout << "Out.\n";

            double error = cross_entropy_error.get_error(train_labels.get_tensor().block(0, 0, 16, 1), out_dense);

            std::cout << " " << error;

            grads = cross_entropy_error.get_gradient();
            std::cout << "First block grads.\n";

            grads = dense2.backward(grads, lr);
            std::cout << "Second block grads.\n";

            grads = relu3.backward(grads, lr);
            grads = dense1.backward(grads, lr);
            std::cout << "Third block grads.\n";

            conv_grads = flatten.backward(grads);
            std::cout << "Flatten grads.\n";

            conv_grads = pool2.backward(conv_grads, lr);
            conv_grads.set_tensor(relu2.backward(conv_grads.get_tensor(), lr),
                    conv_grads.get_tensor_height(), conv_grads.get_tensor_width(), conv_grads.get_tensor_depth());
            conv_grads = conv2.backward(conv_grads, lr);
            std::cout << "Fourth block grads.\n";


            conv_grads = pool1.backward(conv_grads, lr);
            conv_grads.set_tensor(relu1.backward(conv_grads.get_tensor(), lr),
                    conv_grads.get_tensor_height(), conv_grads.get_tensor_width(), conv_grads.get_tensor_depth());
            conv_grads = conv1.backward(conv_grads, lr);
            std::cout << "Fifth block grads.\n";


        }
    }



    /*
    libdl::layers::DenseLayer2D dl2d(2, 2, "Test Layer");
    libdl::layers::DenseLayer2D middle(2, 2, "Middle");
    libdl::layers::DenseLayer2D dl2d_1(2, 1, "Test Layer 2");

    libdl::layers::Sigmoid sig1;
    libdl::layers::Sigmoid sig2;
    libdl::layers::Sigmoid sig3;



    MatrixXd input(5, 2);
    input(0, 0) = 0;
    input(0, 1) = 1;

    input(1, 0) = 1;
    input(1, 1) = 1;

    input(2, 0) = 1;
    input(2, 1) = 0;

    input(3, 0) = 0;
    input(3, 1) = 0;

    input(4, 0) = 0;
    input(4, 1) = 0;

    VectorXd labels(5);
    labels(0) = 1;
    labels(1) = 0;
    labels(2) = 1;
    labels(3) = 0;
    labels(4) = 0;

    libdl::error::ErrorFunctions e(1, labels);


    Eigen::MatrixXd out1;
    Eigen::MatrixXd out2;
    Eigen::MatrixXd out3;
    Eigen::MatrixXd grads;

    double alpha = 0.5;

    std::cout << "Error: ";
    for (int i = 0; i < 3000; i++){
        out1 = dl2d.forward(input);
        out1 = sig1.forward(out1);
        //std::cout << "Out layer 1: \n" << out1 << std::endl;


        out2 = middle.forward(out1);
        out2 = sig2.forward(out2);
        //std::cout << "Out layer 2: \n" << out2 << std::endl;

        out3 = dl2d_1.forward(out2);
        out3 = sig3.forward(out3);
        //std::cout << "Out layer 3: \n" << out3 << std::endl;

        auto err = e.get_error(labels, out3);

        if(i % 100 == 0){
            std::cout << err << " ";
        }

        grads = e.get_gradient();

        grads = sig3.backward(grads, alpha);
        grads = dl2d_1.backward(grads, alpha);

        grads = sig2.backward(grads, alpha);
        grads = middle.backward(grads, alpha);

        grads = sig1.backward(grads, alpha);
        grads = dl2d.backward(grads, alpha);

    }

    //Here I test new input points

    Eigen::MatrixXd test_in(5, 2);

    test_in(0, 0) = 1; // 1
    test_in(0, 1) = 0;

    test_in(1, 0) = 1; // 1
    test_in(1, 1) = 0;

    test_in(2, 0) = 1;
    test_in(2, 1) = 1; // 0

    test_in(3, 0) = 1;
    test_in(3, 1) = 1; // 0

    test_in(4, 0) = 0; // 1
    test_in(4, 1) = 1;

    Eigen::MatrixXd o1;
    Eigen::MatrixXd o2;
    Eigen::MatrixXd o3;

    o1 = dl2d.forward(test_in);
    o1 = sig1.forward(o1);
    o2 = middle.forward(o1);
    o2 = sig2.forward(o2);
    o3 = dl2d_1.forward(o2);
    o3 = sig3.forward(o3);


    std::cout << "\nOutput: \n" << o3 << std::endl;*/



    return 0;
}


/*
SCENARIO("Testing the Convolution Layer", "[ConvolutionLayer]"){
    GIVEN("A convolution layer"){
        libdl::layers::Convolution2D conv(3, 16, 1, 1, 3);

        libdl::TensorWrapper_Exp input(16, 28, 28, 3);
        input.set_tensor(Eigen::MatrixXd::Constant(16, 28*28*3, 1));


        WHEN("random input goes through the layer"){
            libdl::TensorWrapper_Exp output(16, 28, 28, 16);
            output = conv.forward(input);

            std::cout << "Output slice: " << output.get_slice(0, 0) << std::endl;

        }


    }
}*/

SCENARIO("Testing maxpool layer", "[MaxPool]"){

}