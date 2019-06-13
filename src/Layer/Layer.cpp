//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>
#include <string>

#include "Layer.h"
#include "Eigen/Dense"
#include "spdlog/spdlog.h"
#include <cmath>

using namespace libdl::layers;

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
template <typename Tensor>
Layer<Tensor>::Layer() {

}
template <typename Tensor>
Layer<Tensor>::~Layer() {

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


DenseLayer2D::DenseLayer2D(int input_features, int num_neurons, std::string name="Layer"){
    try {

        this->num_neurons = num_neurons;
        this->weights = std::make_unique<Eigen::MatrixXd>(input_features, this->num_neurons);
        this->biases = std::make_unique<Eigen::VectorXd>(this->num_neurons);
        this->name = name;

        *(this->weights) = Eigen::MatrixXd::Random(input_features, this->num_neurons);
        *(this->biases) = Eigen::VectorXd::Constant(this->num_neurons, 1);

    }catch(std::bad_alloc err){
        std::cerr << "Not enough space in memory for the weight declaration!" << std::endl;
        std::cerr << "Layer: " << this->name << std::endl;
        std::exit(-1);
    }

}

std::vector<Eigen::MatrixXd> DenseLayer2D::forward(std::vector<Eigen::MatrixXd> input) {
/*
    try{

        this->input = std::make_unique<Eigen::MatrixXd>(input);

        if(this->weights->rows() != input.cols()) {
            std::string msg;
            msg = "Not compatible shapes: " + std::to_string(this->weights->rows()) + " != " +
                    std::to_string(input.cols()) + " !";
            throw msg;
        }

        Eigen::MatrixXd temp;
        temp = input * *(this->weights);

        temp.rowwise() += this->biases->transpose();


        return temp;

    }catch (const std::string msg){
        std::cerr << msg <<std::endl;
        std::exit(-1);
    }catch(...){
        std::cerr << "Unexpected error happend in the forward pass of layer: " << this->name <<std::endl;
        std::exit(-1);
    }*/
    return input;
}

std::vector<Eigen::MatrixXd> DenseLayer2D::backward(std::vector<Eigen::MatrixXd> gradient, double lr) {
    /*
    //update weights
    *(this->weights) -= lr * (this->input->transpose() * gradient); // replace 4 by N
    *(this->biases) -= lr * gradient.colwise().sum().transpose();

    return gradient * this->weights->transpose();*/
    return gradient;
}


std::string DenseLayer2D::info(){
    std::string str;

    str = "Shape: " + std::to_string(this->weights->rows()) + "x" + std::to_string(this->weights->cols()) + "\n";

    return str;
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </DenseLayer>                             /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Sigmoid>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

std::vector<Eigen::MatrixXd> Sigmoid::forward(std::vector<Eigen::MatrixXd> input){
    /*
    this->input = std::make_unique<Eigen::MatrixXd>(input);

    return input.unaryExpr([this](double e){ return this->sigmoid(e);});*/
    return input;
}

std::vector<Eigen::MatrixXd> Sigmoid::backward(std::vector<Eigen::MatrixXd> gradient, double lr){
    /*
    //std::cout << "\nShape of temp:\n" << temp.rows() << "x" << temp.cols() << std::endl;
    //std::cout << "\nShape of gradient: \n" << gradient.rows() << "x" << gradient.cols() << std::endl;

    return this->input->unaryExpr([this](double e)
    {return this->sigmoid(e) * (1 - this->sigmoid(e));}).array() * gradient.array();*/
    return gradient;
}

double Sigmoid::sigmoid(double input){
    return 1 / (1 + std::exp(-input));
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Convolution2D>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::layers::Convolution2D::Convolution2D(int kernel_size_, int num_filters_, int stride_, int padding_, int input_depth_):
        kernel_size(kernel_size_), num_filters(num_filters_), stride(stride_), padding(padding_){

    this->filters = std::vector<libdl::TensorWrapper3D>(this->num_filters, libdl::TensorWrapper3D(this->kernel_size,
            this->kernel_size, this->input_depth));

    for(int filter = 0; filter < this->num_filters; filter++){

        for(int filter_slice = 0; filter_slice < this->input_depth; filter_slice++){
            this->filters[filter].at(filter_slice) = Eigen::MatrixXd::Random(this->kernel_size, this->kernel_size);//Double check if this changes the weights for real
            //it might be returning a copy to the real thing
        }
    }

    this->biases = std::make_unique<Eigen::VectorXd>(this->num_filters);
    *(this->biases) = Eigen::VectorXd::Constant(this->num_filters, 1);
    //this->filters = this->weights;

}


std::vector<libdl::TensorWrapper3D> libdl::layers::Convolution2D::forward(std::vector<libdl::TensorWrapper3D> inputs_){//this should be multiple 2D images
    this->input = std::vector<libdl::TensorWrapper3D>(inputs_.size(), libdl::TensorWrapper3D(
            inputs_[0].get_first_dim(), inputs_[0].get_second_dim(), inputs_[0].get_third_dim()));
    this->input = inputs_;

    int o_rows = ((this->input[0].get_first_dim() + (2 * this->padding) - this->kernel_size)/this->stride) + 1; //same for all instances
    int o_cols = (this->input[0].get_second_dim() + (2 * this->padding) - this->kernel_size)/this->stride + 1; //same for all instances

    std::vector<libdl::TensorWrapper3D> res(this->input.size(), libdl::TensorWrapper3D(o_rows, o_cols, this->num_filters));
    Eigen::MatrixXd temp;

    for(int instance = 0; instance < this->input.size(); instance++){//iterate through instances
        for(int filter = 0; filter < this->num_filters; filter++) {//iterate through filters

            for (int slice = 0; slice < this->filters[0].get_third_dim(); slice++){ //slices number should be the same for filter and image
                temp = this->correlation2D(this->input.at(instance)(slice), this->filters.at(filter)(slice));
            }

            res.at(instance)(filter) += temp;
        }
    }

    return res;
}

std::vector<libdl::TensorWrapper3D> libdl::layers::Convolution2D::backward(std::vector<libdl::TensorWrapper3D> gradients_, double lr){//Multiple 2D gradients
    return gradients_;
}


//template <typename Tensor>
Eigen::MatrixXd libdl::layers::Convolution2D::correlation2D(Eigen::MatrixXd to_corralate_,Eigen::MatrixXd filter_) const{
    //*(this->input) = input;

    //add padding if there is padding to be added

    int o_rows = ((to_corralate_.rows() + (2 * this->padding) - this->kernel_size)/this->stride) + 1;
    int o_cols = (to_corralate_.cols() + (2 * this->padding) - this->kernel_size)/this->stride + 1;

    this->add_padding2D(to_corralate_);//Working as it should

    std::cout << "\nOutput shape: " << o_rows << "x" << o_cols << std::endl;

    Eigen::MatrixXd output(o_rows, o_cols);

    for(int i = 0; i < o_rows; i++){
        for(int j = 0; j < o_cols; j++){
            output(i, j) = (to_corralate_.block(i, j, this->kernel_size, this->kernel_size).array()*
                    filter_.array()).sum();
        }
    }


    return output;

}

//Maybe this will not be neccessary: Most probably
//template <typename Tensor>
Eigen::MatrixXd libdl::layers::Convolution2D::rotate180(Eigen::MatrixXd filter) {

}

//Adds *padding* rows in each direction.
//template <typename Tensor>
Eigen::MatrixXd libdl::layers::Convolution2D::add_padding2D(Eigen::MatrixXd to_pad_) const{
    //TODO: Implement the padding part
    if(this->padding == 0){
        return to_pad_;
    }else{

        Eigen::MatrixXd tmp(to_pad_.rows()+2 * this->padding, to_pad_.cols() + 2 * this->padding);

        tmp = Eigen::MatrixXd::Constant(to_pad_.rows()+2 * padding, to_pad_.cols() + 2 * padding, 0);

        tmp.block(padding, padding, to_pad_.rows(), to_pad_.cols()) = to_pad_;

        to_pad_ = tmp;



        return to_pad_;
    }
}




////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Convolution2D>                          /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

