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

    //int result = Catch::Session().run(argc, argv);

    std::cout << "it is running\n";

    libdl::TensorWrapper_Exp twe(16, 28, 28, 1, true);

    std::cout << twe.get_tensor();

    std::cout << "\n\n\nSeparation line: \n\n\n";

    std::cout << twe.get_slice(10, 0) << std::endl;







    return 0;
}