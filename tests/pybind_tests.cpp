//
// Created by Aldi Topalli on 2019-06-16.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "Layer.h"
#include "ErrorFunctions.h"


using namespace Eigen;

namespace py = pybind11;

Eigen::MatrixXd print(Eigen::MatrixXd &m){

    std::cout << "Printintg matrix: \n\n" << m << std::endl;

    return m;
}

void test_conv(Eigen::MatrixXd input_gradients, Eigen::MatrixXd filter_gradients, Eigen::Matrix input){
    TensorWrapper input_gradients(input_gradients.rows(), 28, 28, 1);
    TensorWrapper filter_gradients(filter_gradients.rows(), 3, 3, 1);

    TensorWrapper my_input_gradient(input_gradients.rows(), 28, 28, 1);
    TensorWrapper my_filter_gradient(filter_gradients.rows(), 3, 3, 1);
    TensorWrapper input_tensor(input.rows(), 28, 28, 1);
    input_tensor.get_tensor() = input;

    TensorWrapper output(input_gradients.rows(), 26, 26, filter_gradients.rows());

    input_gradients.get_tensor() = input_gradients;
    filter_gradients.get_tensor() = filter_gradients;
    
    libdl::layers::Convolution2D conv("Test conv", 3, 16, 0, 1, 1, 10);

    //Testing starts here

    output = conv.forward(input_tensor);
    
    input_gradients = conv.backward(Matrixd::Constant(input.rows(), output.get_tensor().cols())).
    filter_gradients = conc.get_filter_gradients();

}

void test_pool(){

}

void test_dense(){
    
}



PYBIND11_MODULE(example, m){

    m.def("test_conv", &test_conv, "");
    m.def("test_pool", &test_pool, "");
    m.def("test_dense", &test_pool, "");
}
