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

    /*
    std::cout << "ROTATION IS HAPPENDING HERE.\n";

    Eigen::Transform t(AngleAxisd());

    std::cout << "END OF THE ROTATION.\n";*/

    std::cout << "Downloading data.\n";
    std::unique_ptr<data_handler> dh = std::make_unique<data_handler>();

    dh->read_feature_vector("../data/train-images-idx3-ubyte");
    std::cout << "Downloading features.\n";
    dh->read_feature_label("../data/train-labels-idx1-ubyte");
    std::cout << "Downloading labels.\n";
    dh->split_data();
    dh->count_classes();

    libdl::TensorWrapper_Exp train_data   = dh->convert_training_data_to_Eigen();
    //Leave it here for now but this is not the main cause of those crazy numbers
    for(int i = 0; i < train_data.get_batch_size(); i++){
        double mean = train_data.get_tensor().block(i, 0, 1, train_data.get_tensor().cols()).mean();

        train_data.get_tensor().block(i, 0, 1, train_data.get_tensor().cols()) = train_data.get_tensor().block(i, 0, 1, train_data.get_tensor().cols()).unaryExpr([mean](double e)
        {
            return e - mean;
        });
    }
    std::cout << "Data centered!\n";

    libdl::TensorWrapper_Exp train_labels = dh->convert_training_labels_to_Eigen();
    dh.reset(nullptr);

    int batch_size = 16;
    double lr = 0.001;


    libdl::layers::Convolution2D conv1_1(3, 16, 1, 1, 1); //28x28x1
    libdl::layers::Convolution2D conv1_2(3, 32, 1, 1, 16);
    libdl::layers::ReLU relu1;//28x28x1

    libdl::layers::MaxPool pool1(3, 2);//14x14x16

    libdl::layers::Convolution2D conv2_1(3, 32, 1, 1, 32);//14x14x32
    libdl::layers::Convolution2D conv2_2(3, 64, 1, 1, 32);
    libdl::layers::ReLU relu2;
    libdl::layers::MaxPool pool2(2, 2);//7x7x32

    libdl::layers::Flatten flatten(batch_size, 4, 4, 64);//7x7*32

    libdl::layers::DenseLayer2D dense1(2304, 700, "dense1"); //224x100
    libdl::layers::ReLU relu3;

    libdl::layers::DenseLayer2D dense2(700, 350, "dense2");
    libdl::layers::ReLU relu4;

    libdl::layers::DenseLayer2D dense3(350, 10, "dense3");

    libdl::error::CrossEntropy cross_entropy_error(10);


    libdl::TensorWrapper_Exp batch(batch_size, 28, 28, 1, false);


    libdl::TensorWrapper_Exp out_conv(batch_size, 28, 28, 1, false);
    Eigen::MatrixXd out_dense(batch_size, 4608);

    libdl::TensorWrapper_Exp conv_grads(batch_size, 3, 3, 16, false);
    Eigen::MatrixXd grads(batch_size, 10);



    std::cout << "\nTRAINING PHASE.\n";
    std::cout << "===================================================================\n";

    for(int epoch = 0; epoch < 2; epoch++) {
        for (int b = 0; b < train_data.get_batch_size()/batch_size && b < 50; b++) {
            batch.set_tensor(train_data.get_tensor().block(b, 0, batch_size, 28*28), 28, 28, 1);

            out_conv = conv1_1.forward(batch);
            out_conv = conv1_2.forward(batch);
            out_conv.set_tensor(relu1.forward(out_conv.get_tensor()),
                    out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
            out_conv = pool1.forward(out_conv);

            out_conv = conv2_1.forward(out_conv);
            out_conv = conv2_2.forward(out_conv);
            out_conv.set_tensor(relu2.forward(out_conv.get_tensor()),
                                out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
            out_conv = pool2.forward(out_conv);

            out_dense = flatten.forward(out_conv);

            out_dense = dense1.forward(out_dense);
            out_dense = relu3.forward(out_dense);

            out_dense = dense2.forward(out_dense);
            out_dense = relu4.forward(out_dense);

            out_dense = dense3.forward(out_dense);


            if(b % 10 == 0) {
                double error =
                        cross_entropy_error.get_error(train_labels.get_tensor().block(b, 0, batch_size, 1), out_dense) / 16;
                std::cout << "[Batch: " << b << "; Error: " << error << ";]\n";
            }


            grads = cross_entropy_error.get_gradient();

            grads = dense3.backward(grads, lr);

            grads = relu4.backward(grads, lr);
            grads = dense2.backward(grads, lr);

            grads = relu3.backward(grads, lr);
            grads = dense1.backward(grads, lr);

            conv_grads = flatten.backward(grads);

            conv_grads = pool2.backward(conv_grads, lr);

            conv_grads.set_tensor(relu2.backward(conv_grads.get_tensor(), lr),
                    conv_grads.get_tensor_height(), conv_grads.get_tensor_width(), conv_grads.get_tensor_depth());

            conv_grads = conv2_2.backward(conv_grads, lr);
            conv_grads = conv2_1.backward(conv_grads, lr);

            conv_grads = pool1.backward(conv_grads, lr);
            conv_grads.set_tensor(relu1.backward(conv_grads.get_tensor(), lr),
                    conv_grads.get_tensor_height(), conv_grads.get_tensor_width(), conv_grads.get_tensor_depth());
            conv_grads = conv1_2.backward(conv_grads, lr);
            conv_grads = conv1_1.backward(conv_grads, lr);

        }
    }

    std::cout << "\nTESTING PHASE.\n";
    std::cout << "===================================================================\n";

    for (int b = 50; b < train_data.get_batch_size()/batch_size && b < 82; b++) {
        batch.set_tensor(train_data.get_tensor().block(b, 0, batch_size, 28*28), 28, 28, 1);

        out_conv = conv1_1.forward(batch);
        out_conv = conv1_2.forward(batch);
        out_conv.set_tensor(relu1.forward(out_conv.get_tensor()),
                            out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
        out_conv = pool1.forward(out_conv);

        out_conv = conv2_1.forward(out_conv);
        out_conv = conv2_2.forward(out_conv);
        out_conv.set_tensor(relu2.forward(out_conv.get_tensor()),
                            out_conv.get_tensor_height(), out_conv.get_tensor_width(), out_conv.get_tensor_depth());
        out_conv = pool2.forward(out_conv);

        out_dense = flatten.forward(out_conv);

        out_dense = dense1.forward(out_dense);
        out_dense = relu3.forward(out_dense);

        out_dense = dense2.forward(out_dense);
        out_dense = relu4.forward(out_dense);

        out_dense = dense3.forward(out_dense);


        Vectord predictions = cross_entropy_error.predictions(out_dense, train_labels.get_tensor().block(b, 0, batch_size, 1));

    }


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
