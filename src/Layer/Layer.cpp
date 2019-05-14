//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>

#include "Layer.h"
#include "Eigen/Dense"
#include "spdlog/spdlog.h"

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
template <typename Tensor>
libdl::layers::Layer<Tensor>::Layer() {

}
template <typename Tensor>
libdl::layers::Layer<Tensor>::~Layer() {

}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Layer>                                  /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <DenseLayer>                              /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::layers::DenseLayer2D::DenseLayer2D(int num_neurons): libdl::layers::Layer<Eigen::MatrixXd>::Layer(){
    this->num_neurons = num_neurons;
    this->weights_to_neurons.resize(num_neurons, 1);
}

Eigen::MatrixXd libdl::layers::DenseLayer2D::forward(Eigen::MatrixXd input) {
    try{
        if(input.cols() != this->weights.rows()){
            //throw "Dimensions of the matrix being multiplied do not match: " + input.cols() + " != " + this->weights.rows();
        }
    }catch(const char* msg){
        std::cerr << msg << std::endl;
    }

    return Eigen::MatrixXd::Random(3, 3);
}

Eigen::MatrixXd libdl::layers::DenseLayer2D::backward(Eigen::MatrixXd gradient) {
    return Eigen::MatrixXd::Random(3, 3);
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </DenseLayer>                             /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Perceptron>                              /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


//TODO: Add the functionality for Perceptron layer
//      namely, define only the forward pass for now!
Eigen::MatrixXd libdl::layers::Perceptron::forward(Eigen::MatrixXd input){
    //Eigen::MatrixXd temp = this->weights.dot(input);
    //temp = temp + this->biases;
    return Eigen::MatrixXd::Constant(3, 3, 1);
}

Eigen::MatrixXd libdl::layers::Perceptron::backward(Eigen::MatrixXd gradient){
    //TODO: Implement the backward pass for this function
    return Eigen::MatrixXd::Constant(3, 3, 1.2);
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Perceptron>                             /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Sigmoid>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd libdl::layers::Sigmoid::forward(Eigen::MatrixXd input){
    return Eigen::MatrixXd::Constant(3, 3, 1);
}

Eigen::MatrixXd libdl::layers::Sigmoid::backward(Eigen::MatrixXd gradients){
    return Eigen::MatrixXd::Constant(3, 3, 1);
}

double libdl::layers::Sigmoid::sigmoid(double input){
    return 2;
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

