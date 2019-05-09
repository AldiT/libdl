//
// Created by Aldi Topalli on 2019-05-08.
//

#define CATCH_CONFIG_RUNNER

#include <iostream>
#include "catch.hpp"
#include "Eigen/Dense"

using namespace Eigen;


int main(int argc, char* argv[]){

    int result = Catch::Session().run(argc, argv);

    return result;
}