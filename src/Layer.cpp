//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>

#include "Layer.h"
#include "Eigen/Dense"
#include "../include/Layer.h"


libdl::Layer::Layer() {

}

libdl::Layer::~Layer() {

}


libdl::DenseLayer::DenseLayer(): libdl::Layer::Layer(){

}

Eigen::MatrixXd libdl::DenseLayer::forward() {
    return Eigen::MatrixXd::Random(3, 3);
}

Eigen::MatrixXd libdl::DenseLayer::backward() {
    return Eigen::MatrixXd::Random(3, 3);
}

