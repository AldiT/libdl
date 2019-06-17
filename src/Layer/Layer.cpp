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

Eigen::MatrixXd DenseLayer2D::forward(Eigen::MatrixXd input) {

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
    }
}

Eigen::MatrixXd DenseLayer2D::backward(Eigen::MatrixXd gradient, double lr) {

    //update weights
    *(this->weights) -= lr * (this->input->transpose() * gradient); // replace 4 by N
    *(this->biases) -= lr * gradient.colwise().sum().transpose();

    gradient = gradient * this->weights->transpose();

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

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd input){

    this->input = std::make_unique<Eigen::MatrixXd>(input);

    return  input.unaryExpr([this](double e){ return this->sigmoid(e);});
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd gradient, double lr){

    //std::cout << "\nShape of temp:\n" << temp.rows() << "x" << temp.cols() << std::endl;
    //std::cout << "\nShape of gradient: \n" << gradient.rows() << "x" << gradient.cols() << std::endl;

    return this->input->unaryExpr([this](double e)
                           {return this->sigmoid(e) * (1 - this->sigmoid(e));}).array() * gradient.array();
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


    //For now only stride 1 works
    this->stride = 1;

    this->filters = std::make_unique<libdl::TensorWrapper_Exp>(this->num_filters,
            this->kernel_size, this->kernel_size, this->input_depth, true);//Initialize filters randomly
   //this->filters->set_tensor();

    this->biases = std::make_unique<Eigen::VectorXd>(this->num_filters);
    *(this->biases) = Eigen::VectorXd::Constant(this->num_filters, 1);
    //this->filters = this->weights;

}


libdl::TensorWrapper_Exp libdl::layers::Convolution2D::forward(libdl::TensorWrapper_Exp inputs_){//this should be multiple 2D images

    this->input = std::make_unique<libdl::TensorWrapper_Exp>(inputs_); //operator=

    int o_rows = (this->input->get_tensor_height() + 2 * this->padding - this->kernel_size)/this->stride + 1;
    int o_cols = (this->input->get_tensor_width() + 2 * this->padding - this->kernel_size)/this->stride + 1;

    this->output = std::make_unique<libdl::TensorWrapper_Exp>(this->input->get_batch_size(), o_rows,
            o_cols, this->filters->get_batch_size());

    this->input->correlation(*(this->filters), this->padding, this->stride, *(this->output));

    //TODO: Add biases to the output of the layer


    return *(this->output);
}

libdl::TensorWrapper_Exp libdl::layers::Convolution2D::backward(libdl::TensorWrapper_Exp gradients_, double lr){//Multiple 2D gradients

    libdl::TensorWrapper_Exp filter_gradients;

    filter_gradients = this->input->correlation(gradients_, this->padding, this->stride, filter_gradients);

    filter_gradients = filter_gradients * lr;

    *(this->filters) = *(this->filters) + filter_gradients;

    //TODO: Update the biases aswell, calculate the gradients for biases as well

    gradients_ = gradients_.full_convolution(*(this->filters), gradients_);

    return gradients_;
}


//Maybe this will not be neccessary: Most probably
//template <typename Tensor>
Eigen::MatrixXd libdl::layers::Convolution2D::rotate180(Eigen::MatrixXd filter) {
    return filter;
}

//Adds *padding* rows in each direction.
//template <typename Tensor>





////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Convolution2D>                          /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

