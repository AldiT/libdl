//
// Created by Aldi Topalli on 2019-05-08.
//

#define CATCH_CONFIG_RUNNER

#include <iostream>
#include "catch.hpp"
#include "Eigen/Dense"
#include "TensorWrapper_tests.cpp"
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"
#include "Layer.h"

using namespace Eigen;

namespace py = pybind11;

int main(int argc, char* argv[]){

    //int result = Catch::Session().run(argc, argv);

    libdl::layers::DenseLayer2D dl2d(2, 3);

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

    std::cout << dl2d.get_weights() << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << dl2d.forward(input) << std::endl;


    return 0;
}