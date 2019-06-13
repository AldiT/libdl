//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSOR_TENSORWRAPPER_H
#define LIBDL_TENSOR_TENSORWRAPPER_H

#include <iostream>
#include "TensorWrapper.h"
#include "Eigen/Dense"
#include <list>
#include <vector>
#include <cstdarg>


typedef Eigen::MatrixXd DynMatrix;

using namespace libdl;

TensorWrapper3D::TensorWrapper3D(int first_dim_, int second_dim_, int third_dim_):
                        first_dim(first_dim_), second_dim(second_dim_), third_dim(third_dim_)
{
    this->tensor = std::vector<Eigen::MatrixXd>(this->third_dim, Eigen::MatrixXd(this->first_dim, this->second_dim));

    /*
    for (int i = 0; i < this->third_dim; i++){
        this->tensor.push_back(Eigen::MatrixXd::Constant(this->first_dim, this->second_dim, 0));
    }*/
}

TensorWrapper3D::TensorWrapper3D(const libdl::TensorWrapper3D &rhs) {
    this->tensor = rhs.tensor;//TODO: Check how to use std::copy operator on this copy op

    this->first_dim = rhs.get_first_dim();
    this->second_dim = rhs.get_second_dim();
    this->third_dim = rhs.get_third_dim();
}

TensorWrapper3D& TensorWrapper3D::operator=(const TensorWrapper3D& rhs){
    this->tensor = rhs.tensor;//TODO: Check how to use std::copy operator on this copy op

    this->first_dim = rhs.get_first_dim();
    this->second_dim = rhs.get_second_dim();
    this->third_dim = rhs.get_third_dim();
}


TensorWrapper3D::~TensorWrapper3D() {
    //TODO: eplace this temporary work out with smart pointers
    //TODO: smart pointers already handling this, add log functionality here.
}


Eigen::MatrixXd TensorWrapper3D::operator()(int i) {
    try {

        return this->tensor.at(i);
    }catch(std::out_of_range e){
        std::cerr << "TensorWrapper3D says: Indexing out of range, terminating program..." << std::endl;
        std::exit(-1);
    }
}

Eigen::VectorXd TensorWrapper3D::operator()(int i, int j) {

    if((i > this->first_dim || i < 0) || (j > this->second_dim || j < 0)){
        std::cerr << "Indexing out of range, terminating program..." << std::endl;
        std::exit(-1);
    }

    return this->tensor.at(i).row(j);
}

double TensorWrapper3D::operator()(int i, int j, int z) {

    if((i > this->first_dim || i < 0) || (j > this->second_dim || j < 0) || (z > this->third_dim || z < 0)){
        std::cerr << "Indexing out of range, terminating program..." << std::endl;
        std::exit(-1);
    }

    return this->tensor.at(i)(j, z);
}


Eigen::MatrixXd TensorWrapper3D::at(int i) {
    try {

        return this->tensor.at(i);
    }catch(std::out_of_range e){
        std::cerr << "TensorWrapper3D says: Indexing out of range, terminating program..." << std::endl;
        std::exit(-1);
    }
}

Eigen::VectorXd TensorWrapper3D::at(int i, int j) {

    if((i > this->first_dim || i < 0) || (j > this->second_dim || j < 0)){
        std::cerr << "Indexing out of range, terminating program..." << std::endl;
        std::exit(-1);
    }

    return this->tensor.at(i).row(j);
}

double TensorWrapper3D::at(int i, int j, int z) {

    if((i > this->first_dim || i < 0) || (j > this->second_dim || j < 0) || (z > this->third_dim || z < 0)){
        std::cerr << "Indexing out of range, terminating program..." << std::endl;
        std::exit(-1);
    }

    return this->tensor.at(i)(j, z);
}


double TensorWrapper3D::DotProduct(TensorWrapper3D& tensor1_, TensorWrapper3D& tensor2_){
    if(tensor1_.get_first_dim() != tensor2_.get_first_dim() ||
       tensor1_.get_second_dim() != tensor2_.get_second_dim() ||
       tensor1_.get_third_dim() != tensor2_.get_third_dim()){
        std::cerr << "Dimensions do not match, terminating program..." << std::endl;
        std::exit(-1);
    }

    TensorWrapper3D temp(tensor1_.get_first_dim(), tensor1_.get_second_dim(), tensor1_.get_third_dim());

    double res = 0.0;

    for(int i = 0; i < temp.get_third_dim(); i++){
        temp(i) = tensor1_(i).array() * tensor2_(i).array();
        res += temp(i).sum();
    }


    return res;
}

TensorWrapper3D TensorWrapper3D::ElementWiseMult(TensorWrapper3D& tensor1_, TensorWrapper3D& tensor2_){
    if(tensor1_.get_first_dim() != tensor2_.get_first_dim() ||
       tensor1_.get_second_dim() != tensor2_.get_second_dim() ||
       tensor1_.get_third_dim() != tensor2_.get_third_dim()){
        std::cerr << "Dimensions do not match, terminating program..." << std::endl;
        std::exit(-1);
    }

    TensorWrapper3D res(tensor1_.get_first_dim(), tensor1_.get_second_dim(), tensor1_.get_third_dim());


    for(int i = 0; i < res.get_third_dim(); i++){
        res(i) = tensor1_(i).array() * tensor2_(i).array();
    }

    return res;
}


TensorWrapper3D TensorWrapper3D::operator* (TensorWrapper3D& tensor_){
    return TensorWrapper3D::ElementWiseMult(*(this), tensor_);
}

TensorWrapper3D TensorWrapper3D::operator* (double num){
    for(int i = 0; i < this->get_third_dim(); i++){
        this->tensor.at(i) *= num;
    }
}

TensorWrapper3D TensorWrapper3D::operator+ (TensorWrapper3D& tensor_){
    if(this->get_first_dim() != tensor_.get_first_dim() ||
       this->get_second_dim() != tensor_.get_second_dim() ||
       this->get_third_dim() != tensor_.get_third_dim()){
        std::cerr << "Dimensions do not match, terminating program..." << std::endl;
        std::exit(-1);
    }

    auto temp = *(this);

    for(int i = 0; i < this->get_third_dim(); i++){
        temp(i) = this->tensor.at(i) + tensor_(i);
    }

    return temp;
}


int TensorWrapper3D::get_first_dim () const{
    return this->first_dim;
}

int TensorWrapper3D::get_second_dim () const{
    return this->second_dim;
}

int TensorWrapper3D::get_third_dim () const{
    return this->third_dim;
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

//Experimental TensorWrapper

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <TensorWrapper_Exp>                       /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


libdl::TensorWrapper_Exp::TensorWrapper_Exp(int batch_size_, int tensor_height_, int tensor_width_, int tensor_depth_, bool are_filters_):
    batch_size(batch_size_), tensor_height(tensor_height_), tensor_width(tensor_width_), tensor_depth(tensor_depth_), are_filters(are_filters_)
{
    try {
        this->tensor = std::make_unique<Eigen::MatrixXd>(this->batch_size,
                                                         this->tensor_height * this->tensor_width * this->tensor_depth);

        if (this->are_filters)
            *(this->tensor) = Eigen::MatrixXd::Random(this->batch_size,
                                                      this->tensor_height * this->tensor_width * this->tensor_depth);

    }catch(std::bad_alloc &err){
        std::cout << "TensorWrapper_Exp::TensorWrapper_Exp(...): Not enough memory: " << err.what() << std::endl;
        std::exit(-1);
    }

}

//copy constructor
libdl::TensorWrapper_Exp::TensorWrapper_Exp(const libdl::TensorWrapper_Exp &) {

}

//assignment operator
TensorWrapper_Exp& libdl::TensorWrapper_Exp::operator=(const libdl::TensorWrapper_Exp &) const {

}

TensorWrapper_Exp libdl::TensorWrapper_Exp::correlation(const libdl::TensorWrapper_Exp& filters) const {




}

TensorWrapper_Exp& libdl::TensorWrapper_Exp::pad(int padding_) const {

}

Eigen::MatrixXd libdl::TensorWrapper_Exp::get_slice(int instance_, int depth_) const {
    try{
        if(instance_ < 0 || instance_ > this->batch_size)
            throw std::invalid_argument("instance_");

        if(depth_ < 0 || depth_ > this->tensor_depth)
            throw std::invalid_argument("depth_");


        Eigen::MatrixXd res(this->tensor_height, this->tensor_width);

        for(int row = 0; row < this->tensor_height; row++){
            res.block(row, 0, 1, this->tensor_width) = this->tensor->block(instance_, row*this->tensor_width, 1, this->tensor_width);
        }

        return res;
    }catch(std::invalid_argument & err){
        std::cerr << "TensorWrapper::get_slice: The argument provided \"" << err.what() << "\" is not right!";
        std::exit(-1);
    }catch(std::exception &exp){
        std::cerr << "TensorWrapper::get_slice: An unexpected error happend: " << exp.what() << std::endl;
        std::exit(-1);
    }

}

Eigen::MatrixXd libdl::TensorWrapper_Exp::get_tensor() const{
    return *(this->tensor);
}

int libdl::TensorWrapper_Exp::get_batch_size() const{
    return this->batch_size;
}

int libdl::TensorWrapper_Exp::get_tensor_height() const {
    return this->tensor_height;
}

int libdl::TensorWrapper_Exp::get_tensor_width() const {
    return this->tensor_width;
}

int libdl::TensorWrapper_Exp::get_tensor_depth() const {
    return this->tensor_depth;
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </TensorWrapper_Exp>                      /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////






#endif