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


libdl::layers::DenseLayer::DenseLayer(): libdl::layers::Layer::Layer(){

}

Eigen::MatrixXd libdl::layers::DenseLayer::forward() {
    return Eigen::MatrixXd::Random(3, 3);
}

Eigen::MatrixXd libdl::layers::DenseLayer::backward() {
    return Eigen::MatrixXd::Random(3, 3);
}

