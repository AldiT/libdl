//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_LAYER_H
#define LIBDL_LAYER_H

#define CATCH_CONFIG_MAIN


#include "Eigen/Dense"
#include "Catch.hpp"
#include "pybind11/pybind11.h"

namespace libdl::layers {
    class Layer;
    class DenseLayer;
}

class libdl::layers::Layer {
    public:
        Layer();
        Layer(int){};
        ~Layer();

        //forward pass function
        virtual Eigen::MatrixXd forward() = 0;

        //backward pass function
        virtual Eigen::MatrixXd backward() = 0;

        virtual int printCrap() = 0;


    private:


    protected:
        int num_neurons;
        Eigen::MatrixXd weights_to_neurons;

};


class libdl::layers::DenseLayer: protected libdl::layers::Layer{
public:
    DenseLayer();

    Eigen::MatrixXd forward();
    Eigen::MatrixXd backward();

    int printCrap(){
        std::cout << "This should get printed from the test cases!" << std::endl;
        return 0;
    }
};

TEST_CASE("A Dense Layer"){
    libdl::layers::DenseLayer dl;

    REQUIRE(dl.printCrap() == 0);
}

int add(int i, int j){
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}


#endif //LIBDL_LAYER_H
