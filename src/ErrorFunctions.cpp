//
// Created by Aldi Topalli on 2019-05-19.
//
#include <memory>
#include <iostream>
#include "ErrorFunctions.h"
#include "Eigen/Dense"
#include <cmath>

using namespace libdl::error;

libdl::error::ErrorFunctions::ErrorFunctions(int num_classes, Eigen::VectorXd targets): num_classes(num_classes){
    this->targets = std::make_unique<Eigen::VectorXd>(targets);
}


double libdl::error::ErrorFunctions::get_error(Eigen::VectorXd targets, Eigen::VectorXd logits) {

    Eigen::VectorXd maxVal(4);
    Eigen::MatrixXd::Index maxIndex[4];

    if (logits.rows() != this->targets->rows()){
        std::cerr << "Targets number not the same as logits. " << logits.rows() << " !=  " << this->targets->rows()
        << std::endl;
        std::exit(-1);
    }

    this->logits = std::make_unique<Eigen::VectorXd>(logits);


    for (int j = 0; j < 4; j++){
        maxVal(j) = logits.row(j).maxCoeff(&maxIndex[j]);
    }



    return (*(this->targets) - maxVal).unaryExpr([](double e){ return std::pow(e, 2);}).sum();
}

Eigen::VectorXd libdl::error::ErrorFunctions::get_gradient(){
    auto grad = *(this->targets) - *(this->logits);

    return grad;
}