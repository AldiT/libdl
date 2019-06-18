//
// Created by Aldi Topalli on 2019-05-19.
//
#include <memory>
#include <iostream>
#include "ErrorFunctions.h"
#include "Eigen/Dense"
#include <cmath>

using namespace libdl::error;

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <ErrorFunctions>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::error::ErrorFunctions::ErrorFunctions(int num_classes, Vectord targets): num_classes(num_classes){
    this->targets = std::make_unique<Vectord>(targets);
}


double libdl::error::ErrorFunctions::get_error(Vectord targets, Vectord logits) {

    if (logits.rows() != this->targets->rows()){
        std::cerr << "Targets number not the same as logits. " << logits.rows() << " !=  " << this->targets->rows()
        << std::endl;
        std::exit(-1);
    }


    this->logits = std::make_unique<Vectord>(logits);

    return (*(this->targets) - *(this->logits)).unaryExpr([](double e){ return std::pow(e, 2);}).sum()/2;
}

Vectord libdl::error::ErrorFunctions::get_gradient() {
    return (*(this->logits) - *(this->targets));
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </ErrorFunctions>                         /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <CrossEntropy>                            /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


libdl::error::CrossEntropy::CrossEntropy(int num_classes, Vectord targets) {

}

double libdl::error::CrossEntropy::get_error(Vectord targets, Matrixd logits) {
    try{
        this->logits = std::make_unique<Matrixd>(logits);
        if(this->logits == nullptr)
            throw std::bad_alloc();

        if(targets.rows() != logits.rows())//num of instances should be the same
        {
            throw std::invalid_argument("CrossEntropy::get_error: number of instances does not match!");
        }

        double res = 0;

        //sum over the classes
        for(int instance = 0; instance < logits.rows(); instance++){
            res += std::log(this->softmax(instance)(targets(instance)));
        }

        return -res;
    }catch(std::invalid_argument &err){
        std::cerr << "Invalid argument: " << err.what() << std::endl;
        std::exit(-1);
    }catch(std::bad_alloc &err){
        std::cerr << "Bad alloc: " << err.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }
}

Matrixd libdl::error::CrossEntropy::get_gradient() {
    try{

        Matrixd gradients(this->logits->rows(), this->logits->cols());
        Vectord sums = this->logits->unaryExpr([](double e){return std::exp(e);}).rowwise().sum();

        for(int row = 0; row < gradients.rows(); row++){
            gradients.row(row) /= sums(row);
        }

        for(int instance = 0; instance < this->logits->rows(); instance ++){
            gradients(instance, (*(this->targets))(instance)) -= 1; //loss gradient

            gradients.unaryExpr([this](double e){return e / this->logits->rows();});//Normalization

        }

        return -gradients;

    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }
}

Vectord CrossEntropy::softmax(int instance) {
    double sum = this->logits->block(instance, 0, 1, this->num_classes).sum();

    return this->logits->block(instance, 0, 1, this->num_classes).unaryExpr([this, sum](double e)
    {
        return std::exp(e)/std::exp(sum);
    });
}



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </CrossEntropy>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////