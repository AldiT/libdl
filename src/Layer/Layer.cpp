//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>
#include <string>
#include <chrono>
#include "Layer.h"
#include "Eigen/Dense"
#include "Eigen/Core"
//#include "spdlog/spdlog.h"
#include <cmath>
#include <random>

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


DenseLayer2D::DenseLayer2D(int input_features, int num_neurons, std::string name="Layer", int fan_in=288){
    try {

        this->num_neurons = num_neurons;

        this->weights = std::make_unique<Eigen::MatrixXd>(input_features, this->num_neurons);
        this->biases = std::make_unique<Eigen::VectorXd>(this->num_neurons);
        this->name = name;



        //Eigen::MatrixXd::Random(input_features, this->num_neurons)/10;
        *(this->weights) = this->weights->unaryExpr([num_neurons, input_features](double e){
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<double> normal_dist(0, 0.01);//2/std::sqrt(num_neurons)

            return normal_dist(generator);
        });

        *(this->biases) = Eigen::VectorXd::Constant(this->num_neurons, 0.01);

    }catch(std::bad_alloc err){
        std::cerr << "Not enough space in memory for the weight declaration!" << std::endl;
        std::cerr << "Layer: " << this->name << std::endl;
        std::exit(-1);
    }

}

Eigen::MatrixXd DenseLayer2D::forward(Eigen::MatrixXd& input) {

    try{

        if(this->input == nullptr || input.rows() != this->input->rows())
            this->input = std::make_unique<Eigen::MatrixXd>(input);


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



        //std::cout << "\nTHATS FORWARD PASS\n";
        //std::cout << "Input Layer " << this->name << " weights: \n" << std::endl;

        //std::cout << "Weights avg layer " << this->name << " : " << this->weights->mean() << std::endl;
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

Eigen::MatrixXd DenseLayer2D::backward(Eigen::MatrixXd& gradient, double lr) {

    //update weights
    *(this->weights) -= lr * (this->input->transpose() * gradient); // replace 4 by N
    *(this->biases) -= lr * gradient.colwise().sum().transpose();

    gradient = gradient * this->weights->transpose();

    //std::cout << "\nTHATS BACKWARD PASS\n";
    //std::cout << "Gradients avg layer " << this->name << " : " << gradient.mean() << std::endl;
    //std::cout << "Max weight: " << gradient.maxCoeff() << std::endl;
    //std::cout << "Min weight: " << gradient.minCoeff() << std::endl;

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

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd& input){

    if(this->input == nullptr)
        this->input = std::make_unique<Eigen::MatrixXd>(input);


    *(this->input) = input;

    if(this->output == nullptr)
        this->output = std::make_unique<Eigen::MatrixXd>(input.rows(), input.cols());

    *(this->output) = *(this->input);

    *(this->output) = this->output->unaryExpr([this](double e){ return this->sigmoid(e);});

    return  *(this->output);
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd& gradient, double lr){

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

libdl::layers::Convolution2D::Convolution2D(std::string name_, int kernel_size_, int num_filters_,
        int padding_, int stride_, int input_depth_, int input_neurons_):
       name(name_), kernel_size(kernel_size_), num_filters(num_filters_), stride(stride_), padding(padding_), input_depth(input_depth_){

    this->filters = std::make_unique<libdl::TensorWrapper_Exp>(this->num_filters,
         this->kernel_size, this->kernel_size, this->input_depth, true);//Initialize filters randomly

    double variance = 0.01;
    if(input_neurons_ != 0)
        variance = 2/std::sqrt(input_neurons_);
    std::cout << "The value of variance: " << variance << std::endl;

    this->filters->set_tensor(this->filters->get_tensor().unaryExpr([variance](double e){
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);


            std::normal_distribution<double> normal_dist(0, variance);//2/std::sqrt(input_neurons_)

            return normal_dist(generator);
        }), this->filters->get_tensor_height(), this->filters->get_tensor_width(), this->filters->get_tensor_depth());

    //std::cout << "One slice:\n" << this->filters->get_slice(0, 0) << std::endl;

    this->biases = std::make_unique<Eigen::VectorXd>(this->num_filters);
    *(this->biases) = Eigen::VectorXd::Constant(this->num_filters, 1);
    //this->filters = this->weights;

}


libdl::TensorWrapper_Exp libdl::layers::Convolution2D::forward(libdl::TensorWrapper_Exp& inputs_){//this should be multiple 2D images

    if(this->input == nullptr || inputs_.get_batch_size() != this->input->get_batch_size())
        this->input = std::make_unique<libdl::TensorWrapper_Exp>(inputs_); //operator=
    *(this->input) = inputs_;
    std::cout << "Declared" << std::endl;

    int o_rows = (this->input->get_tensor_height() + 2 * this->padding - this->kernel_size)/this->stride + 1;
    int o_cols = (this->input->get_tensor_width() + 2 * this->padding - this->kernel_size)/this->stride + 1;

    if(this->output == nullptr)
        this->output = std::make_unique<libdl::TensorWrapper_Exp>(this->input->get_batch_size(), o_rows,
                                                                  o_cols, this->filters->get_batch_size());
    std::cout << "Declared2" << std::endl;
    //Padding added
    if(this->padding != 0)
        *(this->input) = this->pad(*(this->input));

    std::cout << "Padded" << std::endl;

    auto start_correlation = std::chrono::system_clock::now();
    *(this->output) = this->input->correlation(*(this->filters), this->stride);
    auto end_correlation = std::chrono::system_clock::now();
    std::cout << "correlated" << std::endl;

    std::chrono::duration<double> correlation_duration = end_correlation-start_correlation;


    return *(this->output);
}

libdl::TensorWrapper_Exp libdl::layers::Convolution2D::backward(libdl::TensorWrapper_Exp& gradients_, double lr){//Multiple 2D gradients

    libdl::TensorWrapper_Exp filter_gradients(this->filters->get_batch_size(), // batch refers to kernel
            this->filters->get_tensor_height(), this->filters->get_tensor_width(),
            this->filters->get_tensor_depth());

    if(this->stride > 1)
        gradients_ = this->dilation(gradients_);
    std::cout << "Before filter_conv\n";

    auto filter_conv_start = std::chrono::system_clock::now();
    filter_gradients = this->filter_conv(gradients_, filter_gradients);
    auto filter_conv_end = std::chrono::system_clock::now();

    std::chrono::duration<double> filter_conv_duration = filter_conv_end-filter_conv_start;
    std::cout << "Before pad\n";
    int temp = this->padding;
    this->padding = this->kernel_size-1;
    gradients_ = this->pad(gradients_);
    this->padding = temp;

    
    std::cout << "Before input_conv\n";
    auto input_conv_start = std::chrono::system_clock::now();
    gradients_ = this->input_conv(gradients_);
    auto input_conv_end = std::chrono::system_clock::now();

    std::chrono::duration<double> input_conv_duration = input_conv_end-input_conv_start;

    filter_gradients.get_tensor() = filter_gradients.get_tensor() * lr;

    this->filters->get_tensor() -= filter_gradients.get_tensor();
    std::cout << "Before cleaning out the gradient\n";
    gradients_ = this->clean_gradient(gradients_);

    return gradients_;
}




TensorWrapper libdl::layers::Convolution2D::filter_conv(TensorWrapper &gradients_, TensorWrapper& filter_gradients) {


    Matrixd temp = Matrixd::Constant(filter_gradients.get_tensor_height(), filter_gradients.get_tensor_width(), 0);

    for(int gradient_slice = 0; gradient_slice < gradients_.get_tensor_depth(); gradient_slice++){

        for(int input_slice = 0; input_slice < this->input->get_tensor_depth(); input_slice++){
            temp = Eigen::MatrixXd::Constant(temp.rows(), temp.cols(), 0);

            for(int instance = 0; instance < gradients_.get_batch_size(); instance++){

                temp += TensorWrapper::correlation2D(
                        this->input->get_slice(instance, input_slice),
                        gradients_.get_slice(instance, gradient_slice), this->padding);

            }

            filter_gradients.update_slice(gradient_slice, input_slice, temp);
        }

    }

    //filter_gradients.get_tensor() /= gradients_.get_batch_size();

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
    rotated_filters = this->reverse_tensor(*(this->filters));

    //reshape the gradient - experimental TODO: This probably should go in a seperate function.
    for(int filter = 0; filter < rotated_filters.get_batch_size(); filter++){
        for(int filter_slice = 0; filter_slice < rotated_filters.get_tensor_depth(); filter_slice++){
            temp.update_slice(filter_slice, filter, rotated_filters.get_slice(filter, filter_slice));
        }
    }

    gradients_ = gradients_.correlation(temp, rotated_filters.get_tensor_height()-1);

    return gradients_;
}

//TODO
TensorWrapper& libdl::layers::Convolution2D::clean_gradient(TensorWrapper& gradients_) {
    int x = this->padding;
    int y = this->padding;

    TensorWrapper copy_gradients = gradients_;
    int rows = this->input->get_tensor_height() - 2 * this->padding;
    int cols = this->input->get_tensor_width() - 2 * this->padding;
    

    Matrixd temp(rows, cols);

    gradients_.set_tensor(Matrixd::Constant(gradients_.get_batch_size(),
            rows*cols*gradients_.get_tensor_depth(), 0), rows, cols,
            this->input->get_tensor_depth());
        
    std::cout << "All declared\n";

    for (int instance = 0; instance < gradients_.get_batch_size(); instance++){
        for(int slice = 0; slice < gradients_.get_tensor_depth(); slice++){

            temp = copy_gradients.get_slice(instance,slice).block(x, y,
                    rows, cols);
            gradients_.update_slice(instance, slice, temp);
        }
    }
    std::cout << "After block\n";



    return gradients_;
}

//Adds *padding* rows in each direction.
//template <typename Tensor>
TensorWrapper& libdl::layers::Convolution2D::pad(TensorWrapper& tensor_){
    try{
        int o_rows  = (tensor_.get_tensor_height() + 2 * this->padding);
        int o_cols = (tensor_.get_tensor_width() + 2* this->padding);

        TensorWrapper temp(tensor_.get_batch_size(), o_rows, o_cols, tensor_.get_tensor_depth());

        temp.set_tensor(Matrixd::Constant(tensor_.get_batch_size(), 
        o_rows * o_cols * tensor_.get_tensor_depth(), 0), o_rows, o_cols, tensor_.get_tensor_depth());

        int tensor_row = 0;
        for(int instance = 0; instance < temp.get_batch_size(); instance++){
            tensor_row = 0;

            for(int row = 0; row < temp.get_tensor().cols(); row += temp.get_tensor_width()){

                if(row / temp.get_tensor_width() < this->padding)//first rows
                    continue;
                else if(row + 2 * this->padding * temp.get_tensor_width() >= temp.get_tensor().cols()){
                    break;
                }
                else if((row + temp.get_tensor_width()) % (temp.get_tensor_height()*temp.get_tensor_width()) == 0){//first rows
                    row += 2 * this->padding * temp.get_tensor_width();
                }

                /* 
                std::cout << "Stats: \n";
                std::cout << "tensor_ cols: " << tensor_.get_tensor().cols() << std::endl;
                std::cout << "temp cols: " << temp.get_tensor().cols() << std::endl;
                std::cout << "Row: " << row << std::endl;
                std::cout << "tensor_row: " << tensor_row << std::endl;
                std::cout << "End of stats.\n";
                */

                if(row == temp.get_tensor().cols() - this->padding * temp.get_tensor_width())//last rows
                    break;

                temp.get_tensor().block(instance, row + this->padding, 1, tensor_.get_tensor_width()) 
                            = tensor_.get_tensor().block(instance, tensor_row, 1, tensor_.get_tensor_width());
                tensor_row += tensor_.get_tensor_width(); //go to next row
                
            }
        }

        tensor_ = temp;

        return tensor_;
    }catch(std::exception &exp){
        std::cout << "Convolution2D::pad: Unexpected error happend: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

//Eigen::seq(row*o_cols, row*o_cols + o_cols-1, this->stride)
//Doesn't work
TensorWrapper& libdl::layers::Convolution2D::dilation(TensorWrapper& tensor_){
    try{//rename the function to dilation
        //TODO: put spaces in between gradient matrix to account for stride in backprop.
        if(this->stride == 1){
            std::cout << "Returned here!" << std::endl;
            return tensor_;
        }

        int o_rows = tensor_.get_tensor_height() + (this->stride-1) * (tensor_.get_tensor_height() - 1);
        int o_cols = tensor_.get_tensor_width() + (this->stride-1) * (tensor_.get_tensor_width() - 1);

        TensorWrapper result(tensor_.get_batch_size(), o_rows, o_cols, tensor_.get_tensor_depth());
        result.get_tensor() = Matrixd::Constant(result.get_batch_size(), 
            o_rows*o_cols*result.get_tensor_depth(), 0);

        for(int instance = 0; instance < tensor_.get_batch_size(); instance++){//for each instance
            for(int row = 0; row < result.get_tensor_height(); row += this->stride){
                result.get_tensor()(instance, Eigen::seq(row*o_cols, row*o_cols + o_cols-1, this->stride))
                    = tensor_.get_tensor().block(instance, row/this->stride, 1, tensor_.get_tensor_height());

            }//end row's for
        }//end instance's for

        tensor_ = result;

        return tensor_;
    }catch(std::exception &exp){
        std::cout << "Convolution2D::pad: Unexpected error happend: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

//TODO
TensorWrapper& libdl::layers::Convolution2D::convolution_operation(){

    try{

        //TODO: (but with low priority) manage the case where the input(or filters or whatever) are nullptr

        //Generate indices




        //multiply for each instance



    }catch(std::exception &err){
        std::cout << "Convolution2D::convolution_operation: Unexpected error happend: " << err.what() << std::endl;
        std::exit(-1);
    }

}

TensorWrapper libdl::layers::Convolution2D::reverse_tensor(TensorWrapper& tensor_){
    try{
        TensorWrapper result(tensor_.get_batch_size(), tensor_.get_tensor_height(), 
            tensor_.get_tensor_width(), tensor_.get_tensor_depth());

        int cols = tensor_.get_tensor().cols();
        int rows = tensor_.get_tensor().rows();

        result.set_tensor(tensor_.get_tensor()(Eigen::seq(rows-1, 0, -1), Eigen::seq(cols-1, 0, -1)),
            tensor_.get_tensor_height(), tensor_.get_tensor_width(), tensor_.get_tensor_depth());
        
        return result;

    }catch(std::exception &err){
        std::cout << "Convolution2D::reverse_tensor: Unexpected error happend: " << err.what() << std::endl;
        std::exit(-1);
    }
}



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

Eigen::MatrixXd libdl::layers::Flatten::forward(libdl::TensorWrapper_Exp& input) {
    this->input->set_tensor(input.get_tensor(), input.get_tensor_height(), input.get_tensor_width(),
            input.get_tensor_depth());

    //std::cout << "Output:\n" << this->input->get_slice(0, 0) << std::endl;

    return input.get_tensor();
}


libdl::TensorWrapper_Exp libdl::layers::Flatten::backward(Eigen::MatrixXd &gradients) {
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


Matrixd libdl::layers::Softmax::forward(Matrixd& input) {
    //input should be a vector with 10 elements


    return input;
}

Matrixd libdl::layers::Softmax::backward(Matrixd& gradient, double lr) {
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

TensorWrapper libdl::layers::MaxPool::forward(TensorWrapper& input) {
    if(this->input == nullptr || input.get_batch_size() != this->input->get_batch_size())
        this->input = std::make_unique<TensorWrapper>(input);

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


    auto max_pool_start = std::chrono::system_clock::now();
    this->max_pooling();
    auto max_pool_end = std::chrono::system_clock::now();

    std::chrono::duration<double> max_pool_duration = max_pool_end-max_pool_start;
    //std::cout << "Max Pool took : " << max_pool_duration.count() << std::endl;

    //std::cout << "Output:\n" << this->output->get_slice(0, 0) << std::endl;
    //std::cout << "Output:\n" << this->output->get_slice(0, 1) << std::endl;


    return *(this->output);

}

TensorWrapper libdl::layers::MaxPool::backward(TensorWrapper& gradient, double) {
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


        auto mp_backprop_start = std::chrono::system_clock::now();

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

        auto mp_backprop_end = std::chrono::system_clock::now();
        std::chrono::duration<double> backprop_duration = mp_backprop_end-mp_backprop_start;
        //std::cout << "Max pool backprop took: " << backprop_duration.count() << std::endl;

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
