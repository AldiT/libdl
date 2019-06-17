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
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace Eigen;




/*

int main(int argc, char* argv[]){
    int width, height, bpp;

    uint8_t* img = stbi_load("../../data/trainingSet/0/img_366.jpg", &width, &height, &bpp, 1);

    stbi_image_free(img);

    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            std::cout << img[i*j];
        }
    }



    std::cout << "Running tests...\n";
    int result = Catch::Session().run(argc, argv);

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


    std::cout << "\nOutput: \n" << o3 << std::endl;



    return 0;
}*/


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