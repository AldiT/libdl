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
#include "Model.h"

#include "TensorWrapper.h"

using namespace Eigen;

using namespace libdl;


int main(int argc, char* argv[]){
    std::cout << "Running tests...\n";
    int result = Catch::Session().run(argc, argv);

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