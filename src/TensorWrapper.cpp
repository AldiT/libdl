//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSOR_TENSORWRAPPER_H
#define LIBDL_TENSOR_TENSORWRAPPER_H

#include <iostream>
#include "TensorWrapper.h"
#include "Eigen/Dense"
#include "Eigen/Geometry"
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

    return *this;
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

    return *this;
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


        if (this->are_filters) {
            srand((unsigned int) time(0));
            *(this->tensor) = Eigen::MatrixXd::Random(this->batch_size,
                                                      this->tensor_height * this->tensor_width * this->tensor_depth) /10;
        }

        this->this_size += sizeof(this);
    }catch(std::bad_alloc &err){
        std::cout << "TensorWrapper_Exp::TensorWrapper_Exp(...): Not enough memory: " << err.what() << std::endl;
        std::exit(-1);
    }



}

//copy constructor

libdl::TensorWrapper_Exp::TensorWrapper_Exp(const libdl::TensorWrapper_Exp &copy_cnstr) {
    this->batch_size    = copy_cnstr.get_batch_size();
    this->tensor_height = copy_cnstr.get_tensor_height();
    this->tensor_width  = copy_cnstr.get_tensor_width();
    this->tensor_depth  = copy_cnstr.get_tensor_depth();
    this->are_filters   = copy_cnstr.is_filter();

    this->tensor = std::make_unique<Eigen::MatrixXd>(this->batch_size,
                                                     this->tensor_height * this->tensor_width * this->tensor_depth);

    *(this->tensor)     = copy_cnstr.get_tensor();
    this->this_size += sizeof(this);
}


//OPERATOR OVERLOADING SECTION

//assignment operator
TensorWrapper_Exp& libdl::TensorWrapper_Exp::operator=(const libdl::TensorWrapper_Exp &obj) {
    this->batch_size    = obj.get_batch_size();
    this->tensor_height = obj.get_tensor_height();
    this->tensor_width  = obj.get_tensor_width();
    this->tensor_depth  = obj.get_tensor_depth();
    this->are_filters   = obj.is_filter();

    *(this->tensor)     = obj.get_tensor();

    return *this;
}

const TensorWrapper_Exp libdl::TensorWrapper_Exp::operator+(TensorWrapper_Exp& add_) const{
    try{
        if(add_.get_tensor().rows() != this->tensor->rows() || add_.get_tensor().cols() != this->tensor->cols()){
            throw std::invalid_argument("The size of the tensors does not match!"); //TODO: Maybe check individual dimensions so that you know which one to set to which value
        }

        *(this->tensor) = *(this->tensor) + add_.get_tensor();

        return *this;

    }catch(std::invalid_argument &err){
        std::cerr << "libdl::TensorWrapper::operator+: " << err.what() << std::endl;
        std::exit(-1);
    }
}

const TensorWrapper_Exp libdl::TensorWrapper_Exp::operator*(double weight) const{
    auto temp = *(this->tensor) * weight;

    TensorWrapper_Exp res(*this);
    res.set_tensor(temp, this->tensor_height, this->tensor_width, this->tensor_depth);

    return res;
}

TensorWrapper_Exp& libdl::TensorWrapper_Exp::correlation(libdl::TensorWrapper_Exp& filters, int padding, int stride) {
    try {


        int o_rows = ((this->get_tensor_height() + (2 * padding) - filters.get_tensor_height())/stride) + 1;
        int o_cols = (this->get_tensor_width() + (2 * padding) - filters.get_tensor_width())/stride + 1;


        libdl::TensorWrapper_Exp *output = new libdl::TensorWrapper_Exp(
                this->get_batch_size(), o_rows, o_cols, filters.get_batch_size());

        output->set_tensor(Eigen::MatrixXd::Constant(this->get_batch_size(), o_rows*o_cols*filters.get_batch_size(), 0),
                o_rows, o_cols, filters.get_batch_size());

        libdl::TensorWrapper_Exp padded_tensor(this->batch_size, this->tensor_height+2*padding,
                this->tensor_width+2*padding, this->tensor_depth);


        Eigen::MatrixXd temp(o_rows, o_cols);

        //std::cout << "Before the for loops.\n";
        //std::cout << "This shape: " << this->tensor_height << "x" << this->tensor_width << std::endl;


        for (int instance = 0; instance < this->get_batch_size(); instance++) {
            for (int filter = 0; filter < filters.get_batch_size(); filter++) {

                temp = Eigen::MatrixXd::Constant(o_rows, o_cols, 0);

                //std::cout << "Inner for loop\n";
                for (int slice = 0; slice < this->get_tensor_depth(); slice++) {
                    //std::cout << "instance_slice\n";
                    auto instance_slice = this->get_slice(instance, slice);
                    //std::cout << "filter_slice\n";
                    auto filter_slice = filters.get_slice(filter, slice);

                    //std::cout << "instance_slice shape: " << instance_slice.rows() << "x" << instance_slice.cols() << std::endl;

                    padded_tensor.update_slice(instance, slice, libdl::TensorWrapper_Exp::pad(instance_slice, padding));
                    auto respond = libdl::TensorWrapper_Exp::correlation2D(padded_tensor.get_slice(instance, slice),
                            filter_slice, 0, stride);
                    temp = temp + respond;

                }
                //std::cout << "Updating output.\n";


                output->update_slice(instance, filter, temp);
            }
        }

        //this->set_tensor(padded_tensor.get_tensor(), padded_tensor.get_tensor_height(), padded_tensor.get_tensor_width(),
         //                padded_tensor.get_tensor_depth());
        //std::cout << "Output shape: " << output->get_batch_size() << "x" << output->get_tensor_height()
        //<< "x" << output->get_tensor_width() << "x" << output->get_tensor_depth() << std::endl;
        return *output;
    }catch(std::invalid_argument &err){
        std::cerr << "libdl::TensorWrapper_Exp::correlation: " << err.what() << std::endl;
        std::exit(-1);
    }


}

TensorWrapper_Exp& libdl::TensorWrapper_Exp::full_convolution(libdl::TensorWrapper_Exp &filters) {


    //std::cout << "full_convolution\n";
    //HINT: If you padd the input you can use the samek convolutions as the one written above.


    return this->correlation(filters, filters.get_tensor_height()-1, 1);
}

Eigen::MatrixXd libdl::TensorWrapper_Exp::correlation2D(Eigen::MatrixXd m1, Eigen::MatrixXd kernels, int padding, int stride) {

    int o_rows = ((m1.rows() + (2 * padding) - kernels.rows())/stride) + 1;
    int o_cols = (m1.cols() + (2 * padding) - kernels.cols())/stride + 1;



    libdl::TensorWrapper_Exp::pad(m1, padding);//Working as it should

    Eigen::MatrixXd output(o_rows, o_cols);

    for(int i = 0; i < o_rows; i++){
        for(int j = 0; j < o_cols; j++){
            output(i, j) = (m1.block(i, j, kernels.rows(), kernels.cols()).array()*
                    kernels.array()).sum();
        }
    }


    return output;
}

Eigen::MatrixXd& libdl::TensorWrapper_Exp::pad(Eigen::MatrixXd& to_pad_, int padding) {
    if(padding == 0){
        return to_pad_;
    }else{

        Eigen::MatrixXd tmp(to_pad_.rows()+2 * padding, to_pad_.cols() + 2 * padding);

        tmp = Eigen::MatrixXd::Constant(to_pad_.rows()+2 * padding, to_pad_.cols() + 2 * padding, 0);

        tmp.block(padding, padding, to_pad_.rows(), to_pad_.cols()) = to_pad_;

        to_pad_ = tmp;



        return to_pad_;
    }
}

Eigen::MatrixXd libdl::TensorWrapper_Exp::rotate180(Eigen::MatrixXd filter) {

    Eigen::MatrixXd res(filter.rows(), filter.cols());

    res(0, 0) = filter(2, 2);res(0, 1) = filter(2, 1); res(0, 2) = filter(2, 0);
    res(1, 0) = filter(1, 2);res(1, 1) = filter(1, 1); res(1, 2) = filter(1, 0);
    res(2, 0) = filter(0, 2);res(2, 1) = filter(0, 1); res(2, 2) = filter(0, 0);


    return res;

}


Eigen::MatrixXd libdl::TensorWrapper_Exp::get_slice(int instance_, int depth_) const {
    try{
        if(instance_ < 0 || instance_ > this->batch_size)
            throw std::invalid_argument("instance_");

        if(depth_ < 0 || depth_ >= this->tensor_depth) {
            std::cout << "DEPTH: " << depth_ << " tensor_depth: " << this->tensor_depth << std::endl;
            throw std::invalid_argument("depth_");
        }


        Eigen::MatrixXd res(this->tensor_height, this->tensor_width);


        for(int row = 0; row < this->tensor_height; row++){
            res.row(row) = this->tensor->block(instance_, (row*this->tensor_width + depth_ * (tensor_width*tensor_height)),
                    1, this->tensor_width);
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

void libdl::TensorWrapper_Exp::update_slice(int instance_, int depth_, Eigen::MatrixXd new_slice_) {
    try{

        //std::cout << "Own height: " << this->get_tensor_height() << " new_slice: " << new_slice_.rows() << std::endl;
        if(instance_ < 0 || instance_ > this->batch_size)
            throw std::invalid_argument("instance_");

        if(depth_ < 0 || depth_ > this->tensor_depth)
            throw std::invalid_argument("depth_");

        /*if(new_slice_.rows() != this->get_tensor_height() || new_slice_.cols() != this->get_tensor_width())
            throw std::invalid_argument("new_slice_");*/

        for(int row = 0; row < new_slice_.rows(); row++){
            this->tensor->block(instance_, (row*this->tensor_width + depth_ * (tensor_width*tensor_height)),
                    1, this->tensor_width) = new_slice_.row(row);
        }

    }catch(std::invalid_argument & err){
        std::cerr << "TensorWrapper::update_slice_: The argument provided \"" << err.what() << "\" is not right!";
        std::exit(-1);
    }catch(std::exception &exp){
        std::cerr << "TensorWrapper::update_slice_: An unexpected error happend: " << exp.what() << std::endl;
        std::exit(-1);
    }
}


Eigen::MatrixXd& libdl::TensorWrapper_Exp::get_tensor() const{
    return *(this->tensor);
}

void libdl::TensorWrapper_Exp::set_tensor(Eigen::MatrixXd new_tensor, int height, int width, int depth) {
    this->tensor_height = height;
    this->tensor_width = width;
    this->tensor_depth = depth;

    *(this->tensor) = new_tensor;
}

int libdl::TensorWrapper_Exp::get_batch_size()      const{
    return this->batch_size;
}

int libdl::TensorWrapper_Exp::get_tensor_height()   const {
    return this->tensor_height;
}

int libdl::TensorWrapper_Exp::get_tensor_width()    const {
    return this->tensor_width;
}

int libdl::TensorWrapper_Exp::get_tensor_depth()    const {
    return this->tensor_depth;
}

bool libdl::TensorWrapper_Exp::is_filter()          const{
    return this->are_filters;
}

std::string libdl::TensorWrapper_Exp::shape() {
    std::string res;
    res += std::to_string(this->get_batch_size());
    res += "x";
    res += std::to_string(this->get_tensor_height());
    res += "x";
    res += std::to_string(this->get_tensor_width());
    res += "x";
    res += std::to_string(this->get_tensor_depth());

    return  res;
}



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </TensorWrapper_Exp>                      /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////






#endif