//
// Created by Aldi Topalli on 2019-05-08.
//

//#define CATCH_CONFIG_RUNNER

#include <iostream>
//#include "catch.hpp"
#include "Eigen/Dense"
#include "TensorWrapper_tests.cpp"
#include "Layer.h"
#include "ErrorFunctions.h"

using namespace Eigen;

int main(int argc, char* argv[]){

    //int result = Catch::Session().run(argc, argv);

    libdl::layers::DenseLayer2D dl2d(2, 3, "Test Layer");
    libdl::layers::DenseLayer2D dl2d_1(3, 3, "Test Layer 2");
    libdl::layers::Sigmoid sig;



    MatrixXd input(4, 2);
    input(0, 0) = 0;
    input(0, 1) = 1;

    input(1, 0) = 1;
    input(1, 1) = 1;

    input(2, 0) = 1;
    input(2, 1) = 0;

    input(3, 0) = 0;
    input(3, 1) = 0;

    VectorXd labels(4);
    labels(0) = 1;
    labels(1) = 0;
    labels(2) = 1;
    labels(3) = 0;




    auto out1 = dl2d.forward(input);


    auto out2 = dl2d_1.forward(out1);


    out2 = sig.forward(out2);


    libdl::error::ErrorFunctions e(1, labels);

    MatrixXd::Index maxIndex[4];
    VectorXd maxVal(4);

    for(int i = 0; i < 4; i++){
        maxVal(i) = out2.row(i).maxCoeff(&maxIndex[i]);
    }

    std::cout << "Logits: " << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << out2 << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << "Error" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << e.get_error(labels, maxVal) << std::endl;




    return 0;
}