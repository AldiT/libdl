//
// Created by Aldi Topalli on 2019-05-08.
//

#define CATCH_CONFIG_RUNNER

#include <iostream>
#include "catch.hpp"
#include "Eigen/Dense"
#include "TensorWrapper_tests.cpp"

using namespace Eigen;


int main(int argc, char* argv[]){

    int result = Catch::Session().run(argc, argv);

    return result;
}

SCENARIO("Some test case", "[test]") {
    Eigen::MatrixXd m(2, 2);
    m << 1, 2,
            3, 4;

    GIVEN("Kot") {
        REQUIRE(m.rows() == 2);
    }
    WHEN("kot"){
        m.resize(3, 3);
        REQUIRE(m.rows() == 3);
    }
}