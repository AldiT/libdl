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

Eigen::MatrixXd& DenseLayer2D::forward(Eigen::MatrixXd& input) {

    try{

        if(this->input == nullptr)
            this->input = std::make_unique<Eigen::MatrixXd>(input);
        else
            *(this->input) = input;

        if(this->output == nullptr)
            this->output = std::make_unique<Eigen::MatrixXd>(input.rows(), this->weights->cols());

        if(this->weights->rows() != input.cols()) {
            std::string msg;
            msg = "Not compatible shapes: " + std::to_string(this->weights->rows()) + " != " +
                  std::to_string(input.cols()) + " !";
            throw msg;
        }

        *(this->output) = input * *(this->weights);

        this->output->rowwise() += this->biases->transpose();



        return *(this->output);

    }catch (const std::string msg){
        std::cerr << msg <<std::endl;
        std::exit(-1);
    }catch(...){
        std::cerr << "Unexpected error happend in the forward pass of layer: " << this->name <<std::endl;
        std::exit(-1);
    }
}

Eigen::MatrixXd& DenseLayer2D::backward(Eigen::MatrixXd& gradient, double lr) {

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

Eigen::MatrixXd& Sigmoid::forward(Eigen::MatrixXd& input){

    if(this->input == nullptr)
        this->input = std::make_unique<Eigen::MatrixXd>(input);
    else
        *(this->input) = input;

    if(this->output == nullptr)
        this->output = std::make_unique<Eigen::MatrixXd>(input.rows(), input.cols());

    *(this->output) = *(this->input);

    this->output->unaryExpr([this](double e){ return this->sigmoid(e);});

    return  *(this->output);
}

Eigen::MatrixXd& Sigmoid::backward(Eigen::MatrixXd& gradient, double lr){

    //std::cout << "\nShape of temp:\n" << temp.rows() << "x" << temp.cols() << std::endl;
    //std::cout << "\nShape of gradient: \n" << gradient.rows() << "x" << gradient.cols() << std::endl;

    this->input->unaryExpr([this](double e)
                           {return this->sigmoid(e) * (1 - this->sigmoid(e));}).array() * gradient.array();

    return *(this->input);
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


libdl::TensorWrapper_Exp& libdl::layers::Convolution2D::forward(libdl::TensorWrapper_Exp& inputs_){//this should be multiple 2D images

    if(this->input == nullptr)
        this->input = std::make_unique<libdl::TensorWrapper_Exp>(inputs_); //operator=
    else
        *(this->input) = inputs_;


    int o_rows = (this->input->get_tensor_height() + 2 * this->padding - this->kernel_size)/this->stride + 1;
    int o_cols = (this->input->get_tensor_width() + 2 * this->padding - this->kernel_size)/this->stride + 1;

    if(this->output == nullptr)
        this->output = std::make_unique<libdl::TensorWrapper_Exp>(this->input->get_batch_size(), o_rows,
                                                                  o_cols, this->filters->get_batch_size());


    this->input->correlation(*(this->filters), this->padding, this->stride, *(this->output));

    //TODO: Add biases to the output of the layer


    return *(this->output);
}

libdl::TensorWrapper_Exp& libdl::layers::Convolution2D::backward(libdl::TensorWrapper_Exp& gradients_, double lr){//Multiple 2D gradients

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

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Flatten>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::layers::Flatten::Flatten(int batch_size, int height, int width, int depth){
    this->input = std::make_unique<libdl::TensorWrapper_Exp>(batch_size, height, width, depth, false);
    this->gradient = std::make_unique<libdl::TensorWrapper_Exp>(batch_size, height, width, depth, false);
}

Eigen::MatrixXd& libdl::layers::Flatten::forward(libdl::TensorWrapper_Exp& input) {
    return input.get_tensor();
}


libdl::TensorWrapper_Exp& libdl::layers::Flatten::backward(Eigen::MatrixXd &gradients) {
    this->gradient->set_tensor(gradients,
            this->input->get_tensor_height(), this->input->get_tensor_width(), this->input->get_tensor_depth());

    return *(this->gradient);

}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Flatten>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Softmax>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


Matrixd& libdl::layers::Softmax::forward(Matrixd& input) {
    //input should be a vector with 10 elements


    return input;
}

Matrixd& libdl::layers::Softmax::backward(Matrixd& gradient, double lr) {
    return gradient;
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Softmax>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <MaxPool>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::layers::MaxPool::MaxPool(int kernel, int stride) {
    this->window_size = kernel;
    this->stride = stride;
}

TensorWrapper& libdl::layers::MaxPool::forward(TensorWrapper& input) {
    if(this->input == nullptr)
        this->input = std::make_unique<TensorWrapper>(input);
    else
        *(this->input) = input;

    if(this->past_propagation == nullptr)
        this->past_propagation = std::make_unique<TensorWrapper>(input);

    this->past_propagation->set_tensor(Eigen::MatrixXd::Constant(input.get_batch_size(),
         input.get_tensor_depth()*input.get_tensor_height()*input.get_tensor_width(), 0),
         input.get_tensor_height(), input.get_tensor_width(), input.get_tensor_depth());

    if(this->output == nullptr)
        this->output = std::make_unique<TensorWrapper>(input.get_batch_size(),
            (input.get_tensor_height()-this->window_size)/this->stride + 1,
            (input.get_tensor_width()-this->window_size)/this->stride + 1,
            input.get_tensor_depth(), false);

    Matrixd propagate_temp(input.get_tensor_height(), input.get_tensor_width());

    for(int instance = 0; instance < input.get_batch_size(); instance++){
        for(int depth = 0; depth < input.get_tensor_depth(); depth++){
            auto to_pool = input.get_slice(instance, depth);
            to_pool = this->max_pooling(to_pool, propagate_temp);
            this->output->update_slice(instance, depth, to_pool);
            this->past_propagation->update_slice(instance, depth, propagate_temp);
        }
    }

    return *(this->output);

}

TensorWrapper& libdl::layers::MaxPool::backward(TensorWrapper& gradient, double) {
    try {
        if(this->backward_gradient == nullptr)
            this->backward_gradient = std::make_unique<TensorWrapper>(this->input->get_batch_size(),
                             this->input->get_tensor_height(),
                             this->input->get_tensor_width(),
                             this->input->get_tensor_depth(), false);

        this->backward_gradient->set_tensor(Matrixd::Constant(this->input->get_batch_size(),
           this->input->get_tensor_height() * this->input->get_tensor_width() *
           this->input->get_tensor_depth(), 0), this->input->get_tensor_height(),
           this->input->get_tensor_width(), this->input->get_tensor_depth());

        for (int instance = 0; instance < gradient.get_batch_size(); instance++) {
            for (int depth = 0; depth < gradient.get_tensor_depth(); depth++) {
                int element_count = 0;

                for (int feature = 0; feature < this->backward_gradient->get_tensor().cols(); feature++) {
                    if (this->past_propagation->get_tensor()(instance, feature) == 1) {
                        this->backward_gradient->get_tensor()(instance, feature) = gradient.get_tensor()(instance, element_count);
                        element_count++;
                    }
                }
                if (element_count != gradient.get_tensor().cols()) {
                    std::cout << "\n\nelement_count does not match!\n\n";
                    throw std::exception();
                }
            }
        }

        return *(this->backward_gradient);

    }catch(std::exception &err){
        std::cerr << err.what() << std::endl;
        std::exit(-1);
    }

}

Matrixd libdl::layers::MaxPool::max_pooling(Matrixd to_pool_, Matrixd& propagations_) {
    Matrixd result((to_pool_.rows()-this->window_size)/this->stride + 1,
                   (to_pool_.cols()-this->window_size)/this->stride + 1);

    Matrixd::Index x_index, y_index;

    for(int row = 0; row < result.rows(); row++){
        for (int col = 0; col < result.cols(); col++){
            result(row, col) = to_pool_.block(row*this->stride, col*this->stride,
                    this->window_size, this->window_size).maxCoeff(&x_index, &y_index);

            propagations_(x_index, y_index) = 1;
        }
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </MaxPool>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
