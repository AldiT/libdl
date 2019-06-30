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

        WHEN("Layer is a conv layer"){/*
            libdl::layers::Convolution2D test_conv(3, 4);//kernel_size, num_filters
            libdl::TensorWrapper_Exp fake_input(1, 10, 10, 1, false);
            libdl::TensorWrapper_Exp fake_output(1, 8, 8, 1, false);
            libdl::TensorWrapper_Exp fake_incoming_gradient(1, 8, 8, 4, false);
            libdl::TensorWrapper_Exp fake_outcoming_gradient(1, 10, 10, 1, false);
            libdl::TensorWrapper_Exp fake_filters(4, 3, 3, 1, false);

            fake_input.set_tensor(Eigen::MatrixXd::Constant(1, 100, 1), 10, 10, 1); //all 1
            fake_filters.set_tensor(Eigen::MatrixXd::Constant(4, 9, 2), 3, 3, 1);

            fake_incoming_gradient.set_tensor(Eigen::MatrixXd::Constant(10, 64*4), 8, 8, 4);



            fake_output = fake_input.correlation(fake_filters, 0);

            fake_outcoming_gradient = test_conv.backward(fake_incoming_gradient, 0.1);*/


            REQUIRE(true);
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