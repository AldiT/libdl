//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>

#include "Layer.h"
#include "Eigen/Dense"
#include "Layer.h"


libdl::layers::Layer::Layer() {

}

libdl::layers::Layer::~Layer() {

}


libdl::layers::DenseLayer::DenseLayer(int num_neurons): libdl::layers::Layer::Layer(){
    this->num_neurons = num_neurons;
    this->weights_to_neurons.resize(num_neurons, 1);
}

Eigen::MatrixXd libdl::layers::DenseLayer::forward() {
    return Eigen::MatrixXd::Random(3, 3);
}

Eigen::MatrixXd libdl::layers::DenseLayer::backward() {
    return Eigen::MatrixXd::Random(3, 3);
}

