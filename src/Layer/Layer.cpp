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

        *(this->weights) = Eigen::MatrixXd::Random(input_features, this->num_neurons)/10;
        *(this->biases) = Eigen::VectorXd::Constant(this->num_neurons, 0.01);

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

        //std::cout << "Max weight: " << this->weights->maxCoeff() << std::endl;
        //std::cout << "Min weight: " << this->weights->minCoeff() << std::endl;


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
    *(this->weights) -= lr * (this->input->transpose() * gradient)/gradient.rows(); // replace 4 by N
    *(this->biases) -= lr * gradient.colwise().sum().transpose()/gradient.rows();

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

    *(this->output) = this->output->unaryExpr([this](double e){ return this->sigmoid(e);});

    return  *(this->output);
}

Eigen::MatrixXd& Sigmoid::backward(Eigen::MatrixXd& gradient, double lr){

    //std::cout << "\nShape of temp:\n" << temp.rows() << "x" << temp.cols() << std::endl;
    //std::cout << "\nShape of gradient: \n" << gradient.rows() << "x" << gradient.cols() << std::endl;

    *(this->input) = this->input->unaryExpr([this](double e)
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

libdl::layers::Convolution2D::Convolution2D(int kernel_size_, int num_filters_, int padding_, int stride_, int input_depth_):
        kernel_size(kernel_size_), num_filters(num_filters_), stride(stride_), padding(padding_), input_depth(input_depth_){


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

    //*(this->input) = this->pad(*(this->input));

    int o_rows = (this->input->get_tensor_height() + 2 * this->padding - this->kernel_size)/this->stride + 1;
    int o_cols = (this->input->get_tensor_width() + 2 * this->padding - this->kernel_size)/this->stride + 1;

    if(this->output == nullptr)
        this->output = std::make_unique<libdl::TensorWrapper_Exp>(this->input->get_batch_size(), o_rows,
                                                                  o_cols, this->filters->get_batch_size());

    *(this->output) = this->input->correlation(*(this->filters), this->padding, this->stride);


    //std::cout << "OUTPUT shape conv: " << this->output->shape() << std::endl;

    //TODO: Add biases to the output of the layer

    //std::cout << "Max weight: " << this->filters->get_tensor().maxCoeff() << std::endl;
    //std::cout << "Min weight: " << this->filters->get_tensor().minCoeff() << std::endl;

    return *(this->output);
}

libdl::TensorWrapper_Exp& libdl::layers::Convolution2D::backward(libdl::TensorWrapper_Exp& gradients_, double lr){//Multiple 2D gradients

    libdl::TensorWrapper_Exp filter_gradients(this->filters->get_batch_size(),
            this->filters->get_tensor_height(), this->filters->get_tensor_width(),
            this->filters->get_tensor_depth());

    //TODO: Update the biases aswell, calculate the gradients for biases as well
    //Biases sometimes are not used in the CNNs so check and decide based on the performance

    //if(this->padding != 0)
        //gradients_ = this->clean_gradient(gradients_);

    //std::cout << "Input shape: " << this->input->shape() << std::endl;
    //std::cout << "taking filter grads\n";

    filter_gradients = this->filter_conv(gradients_, filter_gradients);

    //std::cout << "filter: " << filter_gradients.shape() << " f: " << f.shape() << std::endl;
    //std::cout << "taking grads\n";
    gradients_ = this->input_conv(gradients_);


    //std::cout << "updating filters\n";
    //std::cout << "Para: \n" << filter_gradients.get_tensor() << std::endl;
    filter_gradients.get_tensor() = filter_gradients.get_tensor() * lr;
    //std::cout << "Pas :\n" << filter_gradients.get_tensor() << std::endl;

    this->filters->get_tensor()  -= filter_gradients.get_tensor();



    return gradients_;
}
/*
TensorWrapper& libdl::layers::Convolution2D::pad(TensorWrapper & to_pad_) {

    try{
        int o_rows = (to_pad_.get_tensor_height() + 2*this->padding);
        int o_cols = (to_pad_.get_tensor_width() + 2*this->padding);

        TensorWrapper res(to_pad_.get_batch_size(), o_rows, o_cols, to_pad_.get_tensor_depth());

        for (int i = 0; i < to_pad_.get_batch_size(); i++){
            for(int depth = 0; depth < to_pad_.get_tensor_depth(); depth++){

                res.update_slice(i, depth, TensorWrapper::pad(to_pad_.get_slice(i, depth), this->padding));
            }
        }

        to_pad_.set_tensor(res.get_tensor(), o_rows, o_cols, res.get_tensor_depth());

        return to_pad_;
    }catch (std::exception &err){
        std::cerr << "Convolution2D::pad: An unexpected error happend: " << err.what() << std::endl;
        std::exit(-1);
    }

}
*/
TensorWrapper libdl::layers::Convolution2D::filter_conv(TensorWrapper &gradients_, TensorWrapper& filter_gradients) {


    Matrixd temp = Matrixd::Constant(filter_gradients.get_tensor_height(), filter_gradients.get_tensor_width(), 0);

    for(int gradient_slice = 0; gradient_slice < gradients_.get_tensor_depth(); gradient_slice++){

        for(int input_slice = 0; input_slice < this->input->get_tensor_depth(); input_slice++){


            for(int instance = 0; instance < gradients_.get_batch_size(); instance++){
                temp += TensorWrapper::correlation2D(
                        this->input->get_slice(instance, input_slice),
                        gradients_.get_slice(instance, gradient_slice), this->padding);

            }

            filter_gradients.update_slice(gradient_slice, input_slice, temp/gradients_.get_batch_size());
        }

    }


    return filter_gradients;
}

TensorWrapper libdl::layers::Convolution2D::input_conv(TensorWrapper &gradients_) {
    TensorWrapper input_gradients(this->input->get_batch_size(), this->input->get_tensor_height(),
            this->input->get_tensor_width(), this->input->get_tensor_depth());

    libdl::TensorWrapper_Exp temp(this->filters->get_tensor_depth(),
          this->filters->get_tensor_height(), this->filters->get_tensor_width(), this->filters->get_batch_size(), false);

    libdl::TensorWrapper_Exp rotated_filters(this->filters->get_batch_size(), this->filters->get_tensor_height(),
            this->filters->get_tensor_width(), this->filters->get_tensor_depth());

    //rotate filters.
    for(int filter = 0; filter < this->filters->get_batch_size(); filter++){
        for(int filter_slice = 0; filter_slice < this->filters->get_tensor_depth(); filter_slice++){
            rotated_filters.update_slice(filter, filter_slice,
                    libdl::TensorWrapper_Exp::rotate180(this->filters->get_slice(filter, filter_slice)));
        }
    }

    //reshape the gradient - experimental TODO: This probably should go in a seperate function.
    for(int filter = 0; filter < rotated_filters.get_batch_size(); filter++){
        for(int filter_slice = 0; filter_slice < rotated_filters.get_tensor_depth(); filter_slice++){
            temp.update_slice(filter_slice, filter, rotated_filters.get_slice(filter, filter_slice));
        }
    }
    //std::cout << "Gradient shape before op: " << gradients_.shape() << std::endl;
    gradients_ = gradients_.correlation(temp, rotated_filters.get_tensor_height()-1);

    //gradients_ = this->clean_gradient(gradients_);

    /*
    if(gradients_.get_tensor().rows() != this->input->get_tensor().rows() ||
       gradients_.get_tensor().cols() != this->input->get_tensor().cols()){
        std::cout << "gradients shape" << gradients_.shape() << " input shape: " << this->input->shape() << std::endl;
    }*/


    return gradients_;
}


TensorWrapper& libdl::layers::Convolution2D::clean_gradient(TensorWrapper& gradients_) {
    int x = this->input->get_tensor_height()-gradients_.get_tensor_height()-1,
        y = this->input->get_tensor_width()-gradients_.get_tensor_width()-1;

    TensorWrapper copy_gradients = gradients_;

    Matrixd temp(this->input->get_tensor_height(), this->input->get_tensor_width());

    gradients_.set_tensor(Matrixd::Constant(this->input->get_batch_size(),
            this->input->get_tensor_height()*this->input->get_tensor_width()*this->input->get_tensor_depth(), 0),
            this->input->get_tensor_height(), this->input->get_tensor_width(),
            this->input->get_tensor_depth());


    for (int instance = 0; instance < gradients_.get_batch_size(); instance++){
        for(int slice = 0; slice < gradients_.get_tensor_depth(); slice++){

            temp = copy_gradients.get_slice(instance,slice).block(x, y,
                    this->input->get_tensor_height(), this->input->get_tensor_width());
            gradients_.update_slice(instance, slice, temp);
        }
    }


    return gradients_;
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
    this->input->set_tensor(input.get_tensor(), input.get_tensor_height(), input.get_tensor_width(),
            input.get_tensor_depth());

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
        this->past_propagation = std::make_unique<TensorWrapper>(input.get_batch_size(), input.get_tensor_height(),
                input.get_tensor_width(), input.get_tensor_depth());

    this->past_propagation->set_tensor(Eigen::MatrixXd::Constant(input.get_batch_size(),
         input.get_tensor_depth()*input.get_tensor_height()*input.get_tensor_width(), 0),//set this tensor to 0
         input.get_tensor_height(), input.get_tensor_width(), input.get_tensor_depth());

    if(this->output == nullptr)
        this->output = std::make_unique<TensorWrapper>(input.get_batch_size(),
            (input.get_tensor_height()-this->window_size)/this->stride + 1,
            (input.get_tensor_width()-this->window_size)/this->stride + 1,
            input.get_tensor_depth(), false);



    this->max_pooling();

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
            int element_count = 0, index = 0;

            for (int feature = 0; feature < this->past_propagation->get_tensor().cols(); feature++) {
                index ++;

                if ((this->past_propagation->get_tensor())(instance, feature) == 1) {
                    this->backward_gradient->get_tensor()(instance, feature) = gradient.get_tensor()(instance, element_count);
                    element_count++;
                }

            }

            if (element_count != gradient.get_tensor().cols()) {
                std::cout << "element_count: " << element_count << "gradient cols: "
                    << gradient.get_tensor().cols() << std::endl;

                std::cout << "\n\nelement_count does not match!\n\n";
                throw std::exception();
            }
        }

        return *(this->backward_gradient);

    }catch(std::exception &err){
        std::cerr << err.what() << std::endl;
        std::exit(-1);
    }

}

void libdl::layers::MaxPool::max_pooling() {
    try {
        int res_rows = (this->input->get_tensor_height() - this->window_size) / this->stride + 1;
        int res_cols = (this->input->get_tensor_height() - this->window_size) / this->stride + 1;

        Matrixd result(res_rows, res_cols);

        Matrixd temp_propagations = Eigen::MatrixXd::Constant(this->input->get_tensor_height(),
                                                              this->input->get_tensor_width(), 0);

        Matrixd::Index x_index, y_index;
        for (int instance = 0; instance < this->input->get_batch_size(); instance++) {
            for (int depth = 0; depth < this->input->get_tensor_depth(); depth++) {
                temp_propagations = Eigen::MatrixXd::Constant(this->input->get_tensor_height(),
                                                              this->input->get_tensor_width(), 0);
                for (int row = 0; row < result.rows(); row++) {
                    for (int col = 0; col < result.cols(); col++) {
                        result(row, col) = this->input->get_slice(instance, depth).block(row * this->stride,
                                                                                         col * this->stride,
                                                                                         this->window_size,
                                                                                         this->window_size).maxCoeff(
                                                                                                     &x_index, &y_index);

                        temp_propagations(row * this->stride + x_index, col * this->stride + y_index) = 1;
                        //std::cout << " " << x_index << "-" << y_index << " ";
                    }
                }

                //std::cout << "Propagation matrix: " << temp_propagations << std::endl;
                this->output->update_slice(instance, depth, result);
                this->past_propagation->update_slice(instance, depth, temp_propagations);
            }
        }
    }catch(std::exception &err){
        std::cerr << "MaxPool::max_pooling: An unexpected error happend: " << err.what() << std::endl;
        std::exit(-1);
    }

}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </MaxPool>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
