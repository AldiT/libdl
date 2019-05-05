//#define CATCH_CONFIG_MAIN
#include <iostream>
#include <functional>

#include "extern/Catch.hpp"
#include "test.cpp"

#include <Eigen/Dense>
#include <pybind11.h>

using namespace Eigen;
namespace py = pybind11;

int main(int argc, char **argv)
{
    Matrix3d m =  Matrix3d::Random();
    m = (m + Matrix3d::Constant(1.2)) * 50;

    Vector3d v(1, 2, 3);

    std::cout << "m * v" << std::endl << m * v << std::endl;

    return 0;
}

