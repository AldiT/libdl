//
// Created by Aldi Topalli on 2019-06-16.
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

typedef Eigen::MatrixXd Matrix;

namespace py = pybind11;

Matrix print(Matrix &m){

    std::cout << "Printintg matrix: \n\n" << m << std::endl;

    return m;
}

PYBIND11_MODULE(example, m){

    m.def("print", &print, "");
}