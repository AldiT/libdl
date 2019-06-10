//
// Created by Aldi Topalli on 2019-05-08.
//

//#define CATCH_CONFIG_RUNNER

#include <iostream>
//#include "catch.hpp"
#include "Eigen/Dense"
#include "TensorWrapper_tests.cpp"
#include "Layer.h"
#include "ErrorFunctions.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace Eigen;


struct test{
    test(const std::string &name_) : name(name_){}

    const std::string getName() const{return name;}
    void setName(std::string &new_name){name = new_name;}

    std::string name;
};

PYBIND11_MODULE(example, m){
    py::class_<test>(m, "test")
            .def(py::init<const std::string &>())
            .def("setName", &test::setName)
            .def("getName", &test::getName)
            .def_readwrite("name", &test::name)
            .def("__repr__",
                    [](const test &a){
                return "< example.test named " + a.name + ">";
            });


}

int main(int argc, char* argv[]){

    //int result = Catch::Session().run(argc, argv);

    libdl::layers::DenseLayer2D dl2d(2, 2, "Test Layer");
    libdl::layers::DenseLayer2D middle(2, 2, "Middle");
    libdl::layers::DenseLayer2D dl2d_1(2, 1, "Test Layer 2");

    libdl::layers::Sigmoid sig1;
    libdl::layers::Sigmoid sig2;
    libdl::layers::Sigmoid sig3;



    MatrixXd input(4, 2);
    input(0, 0) = 0;
    input(0, 1) = 1;

    input(1, 0) = 1;
    input(1, 1) = 0;

    input(2, 0) = 1;
    input(2, 1) = 1;

    input(3, 0) = 0;
    input(3, 1) = 0;

    VectorXd labels(4);
    labels(0) = 1;
    labels(1) = 1;
    labels(2) = 0;
    labels(3) = 0;

    libdl::error::ErrorFunctions e(1, labels);


    Eigen::MatrixXd out1;
    Eigen::MatrixXd out2;
    Eigen::MatrixXd out3;
    Eigen::MatrixXd grads;

    double alpha = 0.5;

    std::cout << dl2d.info() << " "
              << middle.info() << " "
              << dl2d_1.info() << std::endl;

    std::cout << "Error: ";
    for (int i = 0; i < 3000; i++){
        out1 = dl2d.forward(input);
        out1 = sig1.forward(out1);


        out2 = middle.forward(out1);
        out2 = sig2.forward(out2);

        out3 = dl2d_1.forward(out2);
        out3 = sig3.forward(out3);

        auto err = e.get_error(labels, out3);

        if(i % 100 == 0){
            std::cout << err << " ";
        }

        grads = e.get_gradient();

        grads = sig3.backward(grads, alpha);
        grads = dl2d_1.backward(grads, alpha);

        grads = sig2.backward(grads, alpha);
        grads = middle.backward(grads, alpha);

        grads = sig1.backward(grads, alpha);
        grads = dl2d.backward(grads, alpha);

    }

    std::cout << "\nOutput: \n" << out3 << std::endl;
    return 0;
}