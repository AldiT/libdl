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
#include <pybind11/pybind11.h>
#include "TensorWrapper.h"

namespace py = pybind11;

using namespace Eigen;


int main(int argc, char* argv[]){
    std::cout << "Running tests...\n";
    int result = Catch::Session().run(argc, argv);

    std::cout << "All tests ran!\n";

    return 0;
}

void printImg(double img[]){
    std::cout << Eigen::Map<MatrixXd>(img) << std::endl;
}

PYBIND11_MODULE(libdl, m){
    m.doc() = "This function will read an image";
    m.def("printImg", &printImg, "This function prints a matrix");
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