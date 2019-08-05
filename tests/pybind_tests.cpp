//
// Created by Aldi Topalli on 2019-06-16.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "Layer.h"
#include "ErrorFunctions.h"
#include "Model.h"


using namespace Eigen;

namespace py = pybind11;

Eigen::MatrixXd print(Eigen::MatrixXd &m){

    std::cout << "Printintg matrix: \n\n" << m << std::endl;

    return m;
}

int test_conv(Eigen::MatrixXd input_gradients, Eigen::MatrixXd filter_gradients, Eigen::MatrixXd input){
    TensorWrapper input_grads(input_gradients.rows(), 28, 28, 1);
    TensorWrapper filter_grads(filter_gradients.rows(), 3, 3, 1);

    TensorWrapper my_input_gradient(input_gradients.rows(), 28, 28, 1);
    TensorWrapper my_filter_gradient(filter_gradients.rows(), 3, 3, 1);
    TensorWrapper input_tensor(input.rows(), 28, 28, 1);
    input_tensor.get_tensor() = input;

    TensorWrapper output(input_gradients.rows(), 26, 26, filter_gradients.rows());

    input_grads.get_tensor() = input_gradients;
    filter_grads.get_tensor() = filter_gradients;
    
    libdl::layers::Convolution2D conv("Test conv", 3, 16, 0, 1, 1, 10);

    //Testing starts here
    /* 
    output = conv.forward(input_tensor);
    
    input_grads = conv.backward(Matrixd::Constant(input.rows(), output.get_tensor().cols()), 0.01);
    filter_grads = conc.get_filter_gradients();
    */

   return 0;
}



PYBIND11_MODULE(libdl, m){
    //TensorWrapper binding
    py::class_<libdl::TensorWrapper_Exp>(m, "TensorWrapper")
        .def(py::init<int, int, int, int, bool>())
        .def(py::init<const libdl::TensorWrapper_Exp&>())
        .def("__repr__", [](const libdl::TensorWrapper_Exp &t){return "<libdl.TensorWrapper object>";})
        .def("correlation", &libdl::TensorWrapper_Exp::correlation)
        .def("get_slice", &libdl::TensorWrapper_Exp::get_slice)
        .def("update_slice", &libdl::TensorWrapper_Exp::update_slice)
        .def("get_batch_size", &libdl::TensorWrapper_Exp::get_batch_size)
        .def("get_tensor_height", &libdl::TensorWrapper_Exp::get_tensor_height)
        .def("get_tensor_width", &libdl::TensorWrapper_Exp::get_tensor_width)
        .def("get_tensor_depth", &libdl::TensorWrapper_Exp::get_tensor_depth)
        .def("shape", &libdl::TensorWrapper_Exp::shape)
        .def("get_tensor", &libdl::TensorWrapper_Exp::get_tensor)
        .def("set_tensor", &libdl::TensorWrapper_Exp::set_tensor)
        .def("get_slice", &libdl::TensorWrapper_Exp::get_slice)
        .def("update_slice", &libdl::TensorWrapper_Exp::update_slice)
        .def("is_filter", &libdl::TensorWrapper_Exp::is_filter)
        .def_static("correlation2D", &libdl::TensorWrapper_Exp::correlation2D);

    //Layer binding
    //py::class_<libdl::layers::Layer>(m, "Layer");

    py::class_<libdl::layers::Layer> layer(m, "Layer");
    layer.def(py::init<>());
    layer.def("forward", &libdl::layers::Layer::forward);
    layer.def("backward", &libdl::layers::Layer::backward);
    

    //Dense binding
    
    py::class_<libdl::layers::DenseLayer2D>(m, "DenseLayer", layer)
        .def(py::init<int, int, std::string, int>())
        .def("forward", &libdl::layers::DenseLayer2D::forward)
        .def("backward", &libdl::layers::DenseLayer2D::backward)
        .def("info", &libdl::layers::DenseLayer2D::info)
        .def("get_weights", &libdl::layers::DenseLayer2D::get_weights)
        .def("get_biases", &libdl::layers::DenseLayer2D::get_biases)
        .def("get_name", &libdl::layers::DenseLayer2D::get_name);

    //Sigmoid binding
    py::class_<libdl::layers::Sigmoid>(m, "Sigmoid", layer)
        .def(py::init<>())
        .def("forward", &libdl::layers::Sigmoid::forward)
        .def("backward", &libdl::layers::Sigmoid::backward);


    //Convolution binding
    py::class_<libdl::layers::Convolution2D>(m, "Convolution", layer)
        .def(py::init<std::string, int, int, int, int, int, int>())
        .def("forward", &libdl::layers::Convolution2D::forward)
        .def("backward", &libdl::layers::Convolution2D::backward)
        .def("get_filters", &libdl::layers::Convolution2D::get_filters)
        .def("set_filters", &libdl::layers::Convolution2D::set_filters)
        .def("pad", &libdl::layers::Convolution2D::pad)
        .def("dilation", &libdl::layers::Convolution2D::dilation)
        .def("reverse_tensor", &libdl::layers::Convolution2D::reverse_tensor)
        .def("clean_gradient", &libdl::layers::Convolution2D::clean_gradient)
        .def("detect_illegal_combination", &libdl::layers::Convolution2D::detect_illegal_combination)
        .def("get_filter_gradients", &libdl::layers::Convolution2D::get_filter_gradients)
        .def("filter_conv", &libdl::layers::Convolution2D::filter_conv)
        .def("input_conv", &libdl::layers::Convolution2D::input_conv);

    //Flatten binding
    py::class_<libdl::layers::Flatten>(m, "Flatten", layer)
        .def(py::init<int, int, int, int>())
        .def("forward", &libdl::layers::Flatten::forward)
        .def("backward", &libdl::layers::Flatten::backward);

    //ReLU binding
    py::class_<libdl::layers::ReLU>(m, "ReLU", layer)
        .def(py::init<>())
        .def("forward", &libdl::layers::ReLU::forward)
        .def("backward", &libdl::layers::ReLU::backward);

    //TanH binding
    py::class_<libdl::layers::TanH>(m, "TanH", layer)
        .def(py::init<>())
        .def("forward", &libdl::layers::TanH::forward)
        .def("backward", &libdl::layers::TanH::backward);

    //MaxPool binding
    py::class_<libdl::layers::MaxPool>(m, "MaxPool", layer)
        .def(py::init<int, int>())
        .def("forward", &libdl::layers::MaxPool::forward)
        .def("backward", &libdl::layers::MaxPool::backward);

    //Cross Entropy Binding
    py::class_<libdl::error::CrossEntropy>(m, "CrossEntropy")
        .def(py::init<int>())
        .def("get_gradient", &libdl::error::CrossEntropy::get_gradient)
        .def("predictions", &libdl::error::CrossEntropy::predictions);
    
    //BinaryCrossEntropy binding
    py::class_<libdl::error::BinaryCrossEntropy>(m, "BinaryCrossEntropy")
        .def(py::init<>())
        .def("get_errors", &libdl::error::BinaryCrossEntropy::get_errors)
        .def("get_gradient", &libdl::error::BinaryCrossEntropy::get_gradient);

    //Model bindings
    py::class_<libdl::model::Model>(m, "Model", layer)
        .def(py::init<int, double, double, int, int, std::string, std::string, int>())
        .def("add", &libdl::model::Model::add)
        .def("forward", &libdl::model::Model::forward)
        .def("backward", &libdl::model::Model::backward);
}
