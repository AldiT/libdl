//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSOR_TENSORWRAPPER_H
#define LIBDL_TENSOR_TENSORWRAPPER_H

#include "TensorWrapper.h"
#include "Eigen/Dense"


typedef Eigen::MatrixXd DynMatrix;

using namespace libdl;

TensorWrapper::TensorWrapper(): third_dimension(3) {
    this->tensor = new DynMatrix[3];
}

TensorWrapper::TensorWrapper(int thirdD): third_dimension(thirdD) {

}

TensorWrapper::TensorWrapper(DynMatrix tensor){
    this->tensor = new DynMatrix;
    *(this->tensor) = tensor;
}

TensorWrapper::~TensorWrapper() {
    //TODO: eplace this temporary work out with smart pointers
    if(this->third_dimension == 3){
        delete[] this->tensor;
    }else if(this->third_dimension == 1){
        delete this->tensor;
    }
}


TensorWrapper* TensorWrapper::operator()(int i) {
    TensorWrapper temp(this->tensor[i]);
    return nullptr;
}

//Implement indexing first

//After indexing implement multiplication


/*

TensorWrapper TensorWrapper::operator*(libdl::TensorWrapper *tensor) {
    TensorWrapper result(this->third_dimension);

    for(int i = 0; i < this->third_dimension; i++){
        result
    }
}


*/














#endif