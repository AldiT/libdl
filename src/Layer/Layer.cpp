//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>

#include "Layer.h"
#include "Eigen/Dense"
#include "Layer.h"

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::layers::Layer::Layer() {

}

libdl::layers::Layer::~Layer() {

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

libdl::layers::DenseLayer::DenseLayer(int num_neurons): libdl::layers::Layer::Layer(){
    this->num_neurons = num_neurons;
    this->weights_to_neurons.resize(num_neurons, 1);
}

Eigen::MatrixXd libdl::layers::DenseLayer::forward(Eigen::MatrixXd input) {
    return Eigen::MatrixXd::Random(3, 3);
}

Eigen::MatrixXd libdl::layers::DenseLayer::backward(Eigen::MatrixXd gradient) {
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
    Eigen::MatrixXd temp = this->weights.dot(input);
    temp = temp + this->biases;
    return temp;
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
    return input.unaryExpr(&this->sigmoid);
}

Eigen::MatrixXd libdl::layers::Sigmoid::backward(Eigen::MatrixXd gradients){

}

double libdl::layers::Sigmoid::sigmoid(double input){
    return 1 / (1 + std::exp(-x));
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

