//
// Created by Aldi Topalli on 2019-06-25.
//

#include <iostream>
#include "catch.hpp"
#include "TensorWrapper.h"
#include "Layer.h"

#include "Eigen/Dense"


SCENARIO("Aldis tests", "[Aldi]"){
    GIVEN("Some layer"){
        libdl::layers::Convolution2D conv1("conv1", 3, 3, 0, 1, 2);
        TensorWrapper constant_filters(3, 3, 3, 2);
        constant_filters.set_tensor(Eigen::MatrixXd::Constant(3, 3*3*2, 1), 3, 3, 2);
        conv1.set_filters(constant_filters);

        TensorWrapper input(1, 4, 4, 2);
        input.set_tensor(Eigen::MatrixXd::Constant(1, 4*4*2, 1), 4, 4, 2);

        TensorWrapper output(1, 2, 2, 3);

        WHEN("Testing forward pass"){

            output = conv1.forward(input);

            REQUIRE(output.get_tensor() == Eigen::MatrixXd::Constant(1, 2*2*3, 18));
        }

        WHEN("Testing backward pass, with constant gradients 6"){
            TensorWrapper gradient(1, 2, 2, 3);
            gradient.set_tensor(Eigen::MatrixXd::Constant(1, 2*2*3, 6), 2, 2, 3);


        }

        WHEN("Layer is maxpool"){/* Commented out because doing some tests somewhere else that do not pass here.
            libdl::layers::MaxPool pool1(2, 2);

            Matrixd input(1, 16);
            input << 1, 2, 3, 4,
                     1, 2, 3, 4,
                     1, 2, 3, 4,
                     1, 2, 3, 4;

            Matrixd expected_output(1, 4);
            expected_output << 2, 4,
                                2, 4;

            libdl::TensorWrapper_Exp input_tensor(1, 4, 4, 1);
            input_tensor.set_tensor(input, 4, 4, 1);
            libdl::TensorWrapper_Exp out(1, 2,2, 1);
            out.set_tensor(expected_output, 2, 2, 1);

            REQUIRE(pool1.forward(input_tensor).get_tensor() == out.get_tensor());
            std::cout << pool1.backward(pool1.forward(input_tensor), 0.1).get_tensor() << std::endl;*/
        }

    }

}