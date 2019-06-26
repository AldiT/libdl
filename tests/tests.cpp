//
// Created by Aldi Topalli on 2019-06-25.
//

#include <iostream>
#include "catch.hpp"
#include "TensorWrapper.h"
#include "Layer.h"

#include "Eigen/Dense"

/*
SCENARIO("Aldis tests", "[Aldi]"){
    GIVEN("Some layer"){

        WHEN("Layer is maxpool"){
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
            std::cout << pool1.backward(pool1.forward(input_tensor), 0.1).get_tensor() << std::endl;
        }

    }

}*/