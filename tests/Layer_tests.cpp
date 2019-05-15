//
// Created by Aldi Topalli on 2019-05-08.
//

#define CATCH_CONFIG_RUNNER

#include <iostream>
#include "catch.hpp"
#include "Eigen/Dense"
#include "TensorWrapper_tests.cpp"
#include "Layer.h"

using namespace Eigen;

int main(int argc, char* argv[]){

    //int result = Catch::Session().run(argc, argv);

    libdl::layers::DenseLayer2D dl2d(2, 3, "Test Layer");
    libdl::layers::DenseLayer2D dl2d_1(3, 3, "Test Layer 2");

    Eigen::MatrixXd input(4, 2);
    input(0, 0) = 0;
    input(0, 1) = 1;

    input(1, 0) = 1;
    input(1, 1) = 1;

    input(2, 0) = 1;
    input(2, 1) = 0;

    input(3, 0) = 0;
    input(3, 1) = 0;

    Eigen::MatrixXd labels(4, 1);
    labels(0) = 1;
    labels(1) = 0;
    labels(2) = 1;
    labels(3) = 0;

    auto out1 = dl2d.forward(input);

    std::cout << dl2d.get_name() << ": " << std::endl << out1 << std::endl;

    auto out2 = dl2d_1.forward(out1);

    std::cout << dl2d_1.get_name() << ": "<< std::endl << out2 << std::endl;

    std::cout << "=================" << std::endl;
    std::cout << dl2d.get_biases() << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << dl2d.get_weights() << std::endl;




    return 0;
}