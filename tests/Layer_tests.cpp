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

    VectorXd labels(4);
    labels(0) = 1;
    labels(1) = 1;
    labels(2) = 0;
    labels(3) = 0;

    libdl::error::ErrorFunctions e(1, labels);


    Eigen::MatrixXd out1;
    Eigen::MatrixXd out2;
    Eigen::MatrixXd out3;
    Eigen::MatrixXd grads;

    double alpha = 0.5;

    std::cout << dl2d.info() << " "
              << middle.info() << " "
              << dl2d_1.info() << std::endl;

    std::cout << "Error: ";
    for (int i = 0; i < 4000; i++){
        out1 = dl2d.forward(input);
        out1 = sig1.forward(out1);


        out2 = middle.forward(out1);
        out2 = sig2.forward(out2);

        out3 = dl2d_1.forward(out2);
        out3 = sig3.forward(out3);

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

    Eigen::MatrixXd in(1, 2);

    in(0, 0) = 1; in(0, 1) = 0;

    Eigen::MatrixXd o1;
    Eigen::MatrixXd o2;
    Eigen::MatrixXd o3;

    o1 = dl2d.forward(input);
    o1 = sig1.forward(o1);
    o2 = middle.forward(o1);
    o2 = sig2.forward(o2);
    o3 = dl2d_1.forward(o2);
    o3 = sig3.forward(o3);


    std::cout << "\nOutput: \n" << o3 << std::endl;

    return 0;
}