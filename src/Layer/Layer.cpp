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
        

        this->weights = std::make_unique<TensorWrapper>(input_features, this->num_neurons, 1, 1);
        this->biases = std::make_unique<Eigen::VectorXd>(this->num_neurons);
        this->name = name;



        this->weights->get_tensor() = this->weights->get_tensor().unaryExpr([num_neurons, input_features](double e){
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

TensorWrapper DenseLayer2D::forward(TensorWrapper& input) {

    try{

        if(this->input == nullptr || input.get_tensor().rows() != this->input->get_tensor().rows())
            this->input = std::make_unique<TensorWrapper>(input);

        *(this->input) = input;

        if(this->output == nullptr)
            this->output = std::make_unique<TensorWrapper>(input.get_tensor().rows(), 
                this->weights->get_tensor().cols());

        if(this->weights->get_tensor().rows() != input.get_tensor().cols()) {
            std::string msg;
            msg = "Not compatible shapes: " + std::to_string(this->weights->get_tensor().rows()) + " != " +
                  std::to_string(input.get_tensor().cols()) + " !";
            throw msg;
        }
        
        this->output->get_tensor() = input.get_tensor() * this->weights->get_tensor();

        this->output->get_tensor().rowwise() += this->biases->transpose();


        /* 
            std::cout << "\nTHATS FORWARD PASS\n";
            std::cout << "Input Layer " << this->name << " weights: \n" << std::endl;

            std::cout << "Weights avg layer " << this->name << " : " << this->weights->mean() << std::endl;
            std::cout << "Max weight: " << this->weights->maxCoeff() << std::endl;
            std::cout << "Min weight: " << this->weights->minCoeff() << std::endl;
        */

        return *(this->output);

    }catch (const std::string msg){
        std::cerr << msg <<std::endl;
        std::exit(-1);
    }catch(...){
        std::cerr << "Unexpected error happend in the forward pass of layer: " << this->name <<std::endl;
        std::exit(-1);
    }
}

TensorWrapper DenseLayer2D::backward(TensorWrapper& gradient, double lr) {
    double reg = 0.001;

    this->weights->get_tensor() -= lr * this->input->get_tensor().transpose() * gradient.get_tensor();
    this->weights->get_tensor() = this->weights->get_tensor().unaryExpr([reg](double e){return e - (e * reg); });
    *(this->biases) -= lr * gradient.get_tensor().colwise().sum().transpose();
    *(this->biases) = this->biases->unaryExpr([reg](double e){return e - (e*reg);});

    gradient.get_tensor() = gradient.get_tensor() * this->weights->get_tensor().transpose();
/* 
    std::cout << "\nTHATS BACKWARD PASS\n";
    std::cout << "Gradients avg layer " << this->name << " : " << gradient.mean() << std::endl;
    std::cout << "Max gradient: " << gradient.maxCoeff() << std::endl;
    std::cout << "Min gradient: " << gradient.minCoeff() << std::endl;
    std::cout << "Avg gradient: " << gradient.mean() << std::endl;
    
    std::cout << "Weight stats" << this->name << "\n";
    std::cout << "Max weight: " << this->weights->maxCoeff() << std::endl;
    std::cout << "Min weight: " << this->weights->minCoeff() << std::endl;
    std::cout << "Avg of weights: " << this->weights->mean() << std::endl;
    std::cout << "End of stats\n";
*/
    return gradient;
}


std::string DenseLayer2D::info(){
    std::string str;

    str = "Shape: " + std::to_string(this->weights->get_tensor().rows()) + "x" + std::to_string(this->weights->get_tensor().cols()) + "\n";

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

TensorWrapper Sigmoid::forward(TensorWrapper& input){

    if(this->input == nullptr)
        this->input = std::make_unique<TensorWrapper>(input);


    *(this->input) = input;

    if(this->output == nullptr)
        this->output = std::make_unique<TensorWrapper>(input.get_batch_size(), input.get_tensor_height(), 
            input.get_tensor_width(), input.get_tensor_depth());

    *(this->output) = *(this->input);

    this->output->get_tensor() = this->output->get_tensor().unaryExpr([this](double e){ return this->sigmoid(e);});

    return  *(this->output);
}

TensorWrapper Sigmoid::backward(TensorWrapper& gradient, double lr){

    this->input->get_tensor() = this->input->get_tensor().unaryExpr([this](double e)
                           {return this->sigmoid(e) * (1 - this->sigmoid(e));}).array() * gradient.get_tensor().array();

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

    this->filters = std::make_unique<TensorWrapper>(this->num_filters,
         this->kernel_size, this->kernel_size, this->input_depth, true);//Initialize filters randomly

    double variance = 0.01;
    if(input_neurons_ != 0)
        variance = 2/std::sqrt(input_neurons_);
    std::cout << "The value of variance: " << variance << std::endl;

    this->filters->set_tensor(this->filters->get_tensor().unaryExpr([variance](double e){
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);


            std::normal_distribution<double> normal_dist(0, 0.01);//2/std::sqrt(input_neurons_)

            return normal_dist(generator);
        }), this->filters->get_tensor_height(), this->filters->get_tensor_width(), this->filters->get_tensor_depth());

    this->biases = std::make_unique<Eigen::VectorXd>(this->num_filters);
    *(this->biases) = Eigen::VectorXd::Constant(this->num_filters, 1);

}

TensorWrapper libdl::layers::Convolution2D::forward(TensorWrapper& inputs_){//this should be multiple 2D images
    try{
        

        if(this->input == nullptr || inputs_.get_batch_size() != this->input->get_batch_size())
            this->input = std::make_unique<TensorWrapper>(inputs_); //operator=
        *(this->input) = inputs_;
        //std::cout << "Declared" << std::endl;
        
        
        
        int o_rows = (this->input->get_tensor_height() + 2 * this->padding - this->kernel_size)/this->stride + 1;
        int o_cols = (this->input->get_tensor_width() + 2 * this->padding - this->kernel_size)/this->stride + 1;

        if(this->output == nullptr)
            this->output = std::make_unique<TensorWrapper>(this->input->get_batch_size(), o_rows,
                                                                    o_cols, this->filters->get_batch_size());
        //std::cout << "Declared2" << std::endl;
        //Padding added
        //std::cout << "Layer: " << this->name << std::endl;
        //std::cout << "Input shape normally: " << this->input->shape() << std::endl;
        if(this->padding != 0)
            *(this->input) = this->pad(*(this->input));

        if(this->detect_illegal_combination())
            throw std::invalid_argument("Combination of input size, stride and kernel is illegal!");

        //std::cout << "Shape after padding: " << this->input->shape() << std::endl;
        //std::cout << "End of layer : " << this->name << "\n"; 

        //std::cout << "Padded" << std::endl;

        auto start_correlation = std::chrono::system_clock::now();
        *(this->output) = this->input->correlation(*(this->filters), this->stride);
        auto end_correlation = std::chrono::system_clock::now();
        //std::cout << "correlated" << std::endl;

        //std::cout << "After correlation" << std::endl;
        std::chrono::duration<double> correlation_duration = end_correlation-start_correlation;
        /* 
        std::cout << "Stats about weights on layer " << this->name << std::endl;
        std::cout << "Max weight: " << this->filters->get_tensor().maxCoeff() << std::endl;
        std::cout << "Min weight: " << this->filters->get_tensor().minCoeff() << std::endl;
        std::cout << "Avg weight: " << this->filters->get_tensor().mean() << std::endl;
        std::cout << "End of stats\n";
        */
        //std::cout << "Input shape at forward: " << this->name << " " << this->input->shape() << std::endl;

        return *(this->output);
    }catch(std::invalid_argument &exp){
        std::cerr << "libdl::layers::Convolution2D::forward: " << exp.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &exp){
        std::cerr << "libdl::layers::Convolution2D::forward: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

TensorWrapper libdl::layers::Convolution2D::backward(TensorWrapper& gradients_, double lr){//Multiple 2D gradients

    TensorWrapper filter_gradients(this->filters->get_batch_size(), // batch refers to kernel
            this->filters->get_tensor_height(), this->filters->get_tensor_width(),
            this->filters->get_tensor_depth());

    if(gradients_.get_tensor().rows() != this->output->get_tensor().rows() || 
        gradients_.get_tensor().cols() != this->output->get_tensor().cols()){
        std::cout << "Gradient tensor: " << gradients_.get_tensor().rows() << "x" << gradients_.get_tensor().cols() << std::endl;
        std::cout << "Output tensor: " << this->output->get_tensor().rows() << "x" << this->output->get_tensor().cols() << std::endl;
        std::cout << "GRADIENT SHAPE IS NOT THE SAME AS OUTPUT" << std::endl;
    }
    
    gradients_.set_tensor(gradients_.get_tensor(), this->output->get_tensor_height(), this->output->get_tensor_width(),
        this->output->get_tensor_depth());
    /* 
        std::cout << "Stats about gradients: " << std::endl;
        std::cout << "Mean: " << gradients_.get_tensor().mean() << std::endl;
        std::cout << "Max: " << gradients_.get_tensor().maxCoeff() << std::endl;
        std::cout << "Min" << gradients_.get_tensor().minCoeff() << std::endl;

        std::cout << "end of stats\n";
    */


    if(this->stride > 1)
        gradients_ = this->dilation(gradients_);

    filter_gradients = this->filter_conv(gradients_, filter_gradients);
    gradients_ = this->input_conv(gradients_);

    filter_gradients.get_tensor() = filter_gradients.get_tensor() * lr;

    this->filters->get_tensor() -= filter_gradients.get_tensor();

    if(gradients_.get_tensor().rows() != this->input->get_tensor().rows() || 
        gradients_.get_tensor().cols() != this->input->get_tensor().cols()){
        std::cout << "GRADIENT SHAPE IS NOT THE SAME AS INPUT" << std::endl;
        std::cout << "Shape of input tensor: " << this->input->get_tensor().rows() << "x" << this->input->get_tensor().cols() << std::endl;
        std::cout << "Shape of gradient tensor: " << gradients_.get_tensor().rows() << "x" << gradients_.get_tensor().cols() << std::endl;
    }
    
    gradients_ = this->clean_gradient(gradients_);

    return gradients_;
}


//This function calculates the gradient w.r.t. filters 
TensorWrapper libdl::layers::Convolution2D::filter_conv(TensorWrapper gradients_, TensorWrapper& filter_gradients) {

    Matrixd temp = Matrixd::Constant(filter_gradients.get_tensor_height(), filter_gradients.get_tensor_width(), 0);

    int rows = 1;
    for(int gradient_slice = 0; gradient_slice < gradients_.get_tensor_depth(); gradient_slice++){

        for(int input_slice = 0; input_slice < this->input->get_tensor_depth(); input_slice++){
            temp = Matrixd::Constant(temp.rows(), temp.cols(), 0);

            for(int instance = 0; instance < gradients_.get_batch_size(); instance++){


                temp += TensorWrapper::correlation2D(
                        this->input->get_slice(instance, input_slice),
                        gradients_.get_slice(instance, gradient_slice), 1);
            }

            filter_gradients.update_slice(gradient_slice, input_slice, temp);
        }

    }

    return filter_gradients;
}

//This function calculates the gradient w.r.t. input 
TensorWrapper libdl::layers::Convolution2D::input_conv(TensorWrapper gradients_) {
    TensorWrapper input_gradients(this->input->get_batch_size(), this->input->get_tensor_height(),
            this->input->get_tensor_width(), this->input->get_tensor_depth());

    TensorWrapper temp(this->filters->get_tensor_depth(),
          this->filters->get_tensor_height(), this->filters->get_tensor_width(), this->filters->get_batch_size(), false);

    TensorWrapper rotated_filters(this->filters->get_batch_size(), this->filters->get_tensor_height(),
            this->filters->get_tensor_width(), this->filters->get_tensor_depth());

    //rotate filters.
    rotated_filters = this->reverse_tensor(*(this->filters));

    //reshape the gradient - experimental TODO: This probably should go in a seperate function.
    for(int filter = 0; filter < rotated_filters.get_batch_size(); filter++){
        for(int filter_slice = 0; filter_slice < rotated_filters.get_tensor_depth(); filter_slice++){
            temp.update_slice(filter_slice, filter, rotated_filters.get_slice(filter, filter_slice));
        }
    }

    int t = this->padding;
    this->padding = this->filters->get_tensor_height()-1;
    gradients_ = this->pad(gradients_);
    this->padding = t;

    gradients_ = gradients_.correlation(temp, 1);

    return gradients_;
}

//This function discards extra rows and columns from gradient of layers with padding
TensorWrapper& libdl::layers::Convolution2D::clean_gradient(TensorWrapper& gradients_) {
    if(this->padding == 0)
        return gradients_;
    
    //std::cout << "Shape of gradient inside clean_gradient: " << gradients_.shape() << std::endl;

    int x = this->padding;
    int y = this->padding;
    
    //std::cout << "\nClean gradient: " << this->name << "\n";


    TensorWrapper copy_gradients = gradients_;
    int rows = this->input->get_tensor_height() - 2 * this->padding;
    int cols = this->input->get_tensor_width() - 2 * this->padding;
    

    Matrixd temp(rows, cols);

    gradients_.set_tensor(Matrixd::Constant(gradients_.get_batch_size(),
            rows*cols*gradients_.get_tensor_depth(), 0), rows, cols,
            this->input->get_tensor_depth());
        
    //std::cout << "All declared\n";

    for (int instance = 0; instance < gradients_.get_batch_size(); instance++){
        for(int slice = 0; slice < gradients_.get_tensor_depth(); slice++){
            
            //TEST
            /* 
            std::cout << "Stats:\n";
            std::cout << "Shape of temp: " << temp.rows() << "x" << temp.cols() << std::endl;
            std::cout << "Shape of gradients: " << copy_gradients.shape() << std::endl;
            std::cout << "Where it is going to cut: " << x + rows << " x " << y + cols << std::endl;
            std::cout << "End of stats\n";
            */
            //TEST

            temp = copy_gradients.get_slice(instance,slice).block(x, y,
                    rows, cols);
            gradients_.update_slice(instance, slice, temp);
        }
    }
    //std::cout << "After block\n";



    return gradients_;
}

//pad
TensorWrapper& libdl::layers::Convolution2D::pad(TensorWrapper& tensor_){
    try{
        //std::cout << "Padding: " << this->padding << std::endl;

        int o_rows  = (tensor_.get_tensor_height() + 2 * this->padding);
        int o_cols = (tensor_.get_tensor_width() + 2* this->padding);

        TensorWrapper temp(tensor_.get_batch_size(), o_rows, o_cols, tensor_.get_tensor_depth());
        //std::cout << "\nPadding " << this->name << "\n";

        temp.set_tensor(Matrixd::Constant(tensor_.get_batch_size(), 
        o_rows * o_cols * tensor_.get_tensor_depth(), 0), o_rows, o_cols, tensor_.get_tensor_depth());

        int tensor_row = 0;
        for(int instance = 0; instance < temp.get_batch_size(); instance++){
            tensor_row = 0;

            for(int row = 0; row < temp.get_tensor().cols(); row += temp.get_tensor_width()){
                
                if(row / temp.get_tensor_width() < this->padding){
                    row += (this->padding-1) * temp.get_tensor_width();
                    continue;
                    //first rows
                }else if( (row + this->padding * temp.get_tensor_width()) % (temp.get_tensor_height()*temp.get_tensor_width()) == 0){
                    row += (2*this->padding - 1) * temp.get_tensor_width();
                    continue;
                    //the rest, ending and beginning of next rows
                }


                /* 
                std::cout << "Stats: \n";
                std::cout << "tensor_ cols: " << tensor_.get_tensor().cols() << std::endl;
                std::cout << "temp cols: " << temp.get_tensor().cols() << std::endl;
                std::cout << "Row: " << row << std::endl;
                std::cout << "tensor_row: " << tensor_row << std::endl;
                std::cout << "End of stats.\n";
                */

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

// This function dilates the gradient to support stride in convolutions
TensorWrapper& libdl::layers::Convolution2D::dilation(TensorWrapper& tensor_){
    try{//rename the function to dilation
        //TODO: put spaces in between gradient matrix to account for stride in backprop.
        if(this->stride == 1){
            std::cout << "Returned here!" << std::endl;
            return tensor_;
        }

        //std::cout << "\nDilation " << this->name << "\n";

        int o_rows = tensor_.get_tensor_height() + (this->stride-1) * (tensor_.get_tensor_height() - 1);
        int o_cols = tensor_.get_tensor_width() + (this->stride-1) * (tensor_.get_tensor_width() - 1);

        TensorWrapper result(tensor_.get_batch_size(), o_rows, o_cols, tensor_.get_tensor_depth());
        result.get_tensor() = Matrixd::Constant(result.get_batch_size(), 
            o_rows*o_cols*result.get_tensor_depth(), 0);

        int tensor_row = 0;
        for(int instance = 0; instance < tensor_.get_batch_size(); instance++){//for each instance
            tensor_row = 0;

            for(int row = 0; row < result.get_tensor().cols(); row += this->stride*o_cols){

                

                result.get_tensor()(instance, Eigen::seq(row, row + o_cols-1, this->stride))
                    = tensor_.get_tensor().block(instance, tensor_row, 1, tensor_.get_tensor_height());
                
                tensor_row += tensor_.get_tensor_width();
                
                if((row + o_cols) % (o_rows*o_cols) == 0 && row != 0)
                    row -= (this->stride-1)*o_cols;
            }//end row's for
        }//end instance's for

        tensor_ = result;

        return tensor_;
    }catch(std::exception &exp){
        std::cout << "Convolution2D::pad: Unexpected error happend: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

//This function reverses the tensor to perform operations above (calculate gradient)
TensorWrapper libdl::layers::Convolution2D::reverse_tensor(TensorWrapper& tensor_) const{
    try{
        TensorWrapper result(tensor_.get_batch_size(), tensor_.get_tensor_height(), 
            tensor_.get_tensor_width(), tensor_.get_tensor_depth());

        int cols = tensor_.get_tensor().cols();
        int rows = tensor_.get_tensor().rows();
        
        int frame = tensor_.get_tensor_width()*tensor_.get_tensor_height();

        for(int instance = 0; instance < tensor_.get_batch_size(); instance++){
            for(int row = 0; row < tensor_.get_tensor().cols(); row += frame){
               result.get_tensor().block(instance, row, 1, frame) = 
                    tensor_.get_tensor().block(instance, row, 1, frame).reverse();
                
            }
        }


        //result.set_tensor(tensor_.get_tensor()(Eigen::seq(rows-1, 0, -1), Eigen::seq(cols-1, 0, -1)),
          //  tensor_.get_tensor_height(), tensor_.get_tensor_width(), tensor_.get_tensor_depth());
        
        return result;

    }catch(std::exception &err){
        std::cout << "Convolution2D::reverse_tensor: Unexpected error happend: " << err.what() << std::endl;
        std::exit(-1);
    }
}

//This function detects illegal combinations of stride, filter size and input size
bool libdl::layers::Convolution2D::detect_illegal_combination() const{
    if((this->input->get_tensor_height() - this->filters->get_tensor_height()) % this->stride == 0 &&
       (this->input->get_tensor_width() - this->filters->get_tensor_width()) % this->stride == 0)
       return false; //flase means the combination is ok.
    
    return true;//true means it is illegal
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

//After a redesign in the layer functionality this layer is no longer needed
// Previosly I used it to convert from TensorWrapper to Eigen::Matrix

libdl::layers::Flatten::Flatten(int batch_size, int height, int width, int depth){
    this->input = std::make_unique<TensorWrapper>(batch_size, height, width, depth, false);
    this->gradient = std::make_unique<TensorWrapper>(batch_size, height, width, depth, false);
}

TensorWrapper libdl::layers::Flatten::forward(TensorWrapper& input) {
    this->input->set_tensor(input.get_tensor(), input.get_tensor_height(), input.get_tensor_width(),
            input.get_tensor_depth());

    //std::cout << "Output:\n" << this->input->get_slice(0, 0) << std::endl;

    return input;
}


TensorWrapper libdl::layers::Flatten::backward(TensorWrapper &gradients, double lr) {
    this->gradient->set_tensor(gradients.get_tensor(),
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

//Not needed , I perform softmax in the CrossEntropy class since, it is not a real 
//activation , I thought it would be of no use to have a separate layer
TensorWrapper libdl::layers::Softmax::forward(TensorWrapper& input) {
    //input should be a vector with 10 elements
    return input;
}

TensorWrapper libdl::layers::Softmax::backward(TensorWrapper& gradient, double lr) {
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
    //std::cout << "Size of input: " << this->input->shape() << std::endl;

    if(this->past_propagation == nullptr)
        this->past_propagation = std::make_unique<TensorWrapper>(input.get_batch_size(), input.get_tensor_height(),
                input.get_tensor_width(), input.get_tensor_depth());

    this->past_propagation->set_tensor(Matrixd::Constant(input.get_batch_size(),
         input.get_tensor_depth()*input.get_tensor_height()*input.get_tensor_width(), 0),//set this tensor to 0
         input.get_tensor_height(), input.get_tensor_width(), input.get_tensor_depth());

    if(this->output == nullptr)
        this->output = std::make_unique<TensorWrapper>(input.get_batch_size(),
            (input.get_tensor_height()-this->window_size)/this->stride + 1,
            (input.get_tensor_width()-this->window_size)/this->stride + 1,
            input.get_tensor_depth(), false);


    this->max_pooling();

    //std::cout << "Output:\n" << this->output->get_slice(0, 0) << std::endl;
    //std::cout << "Output:\n" << this->output->get_slice(0, 1) << std::endl;

    /* 
    std::cout << "STATS\n";
    std::cout << "Input shape: " << this->input->shape() << std::endl;
    std::cout << "Past propagation shape: " << this->past_propagation->shape() << std::endl;
    std::cout << "Output shape: " << this->output->shape() << std::endl;
    std::cout << "End of stats\n";
    */

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
           this->input->get_tensor_depth(), 0), 
           this->input->get_tensor_height(), this->input->get_tensor_width(), this->input->get_tensor_depth());


        /* 
        std::cout << "STATS\n";
        std::cout << "Input shape: " << this->input->shape() << std::endl;
        std::cout << "Past propagation shape: " << this->past_propagation->shape() << std::endl;
        std::cout << "Output shape: " << this->output->shape() << std::endl;
        std::cout << "Gradient shape: " << gradient.shape() << std::endl;
        std::cout << "End of stats\n";
        */
       gradient.set_tensor(gradient.get_tensor(), this->output->get_tensor_height(), this->output->get_tensor_width(),
       this->output->get_tensor_depth());

       //std::cout << "Past propagation matrix\n";

       //std::cout << this->past_propagation->get_tensor() << std::endl;
       
        /* std::cout << "Outside for\n";
       std::cout << "Stats\n";
        std::cout << "backward gradient shape: " << this->backward_gradient->shape() << std::endl;
        std::cout << "output shape: " << this->output->shape() << std::endl;
        std::cout << "gradient shape: "<< gradient.shape() << std::endl;
        std::cout << "gradient tensor: " << gradient.get_tensor().rows() << "x" << gradient.get_tensor().cols() << std::endl;
        std::cout << "input shape: " << this->input->shape() << std::endl;
       std::cout << "End of stats\n";*/

    
        Matrixd temp(this->backward_gradient->get_tensor_height(), this->backward_gradient->get_tensor_width());
        Matrixd gradient_slice(gradient.get_tensor_height(), gradient.get_tensor_width());

        for(int instance = 0; instance < this->backward_gradient->get_batch_size(); instance++){
            for(int depth = 0; depth < this->backward_gradient->get_tensor_depth(); depth++){
                temp = Matrixd::Constant(this->backward_gradient->get_tensor_height(), 
                                            this->backward_gradient->get_tensor_width(), 0);

                gradient_slice = gradient.get_slice(instance, depth);
                for (int row = 0; row < gradient.get_tensor_height(); row++) {
                    for (int col = 0; col < gradient.get_tensor_width(); col++) {
                        //std::cout << "Inside for\n";
                        temp.block(row * this->stride, col * this->stride,
                            this->window_size, this->window_size) = this->past_propagation->get_slice(instance, depth).block(row * this->stride, col * this->stride,
                            this->window_size, this->window_size) * gradient_slice(row, col);

                        //std::cout << "Product: \n" << this->past_propagation->get_slice(instance, depth).block(row * this->stride, col * this->stride,
                        //    this->window_size, this->window_size) * gradient.get_slice(instance, depth)(row, col) << std::endl;
                        //std::cout << "After this\n";
                    }
                }
                
                this->backward_gradient->update_slice(instance, depth, temp);
            }
        } 

        //std::cout << "Original gradient:\n" << gradient.get_slice(0, 0) << std::endl;
        //std::cout << "Gradient after pool:\n " << this->backward_gradient->get_slice(0, 0) << std::endl;
        
        //std::cout << "Before returning\n";
        //std::cout << "Max pool backprop took: " << backprop_duration.count() << std::endl;

        return *(this->backward_gradient);

    }catch(std::exception &err){
        std::cerr << err.what() << std::endl;
        std::exit(-1);
    }

}

//Performs the max pooling operation , and also stores the mask for back-prop
void libdl::layers::MaxPool::max_pooling() {
    try {
        int res_rows = (this->input->get_tensor_height() - this->window_size) / this->stride + 1;
        int res_cols = (this->input->get_tensor_height() - this->window_size) / this->stride + 1;

        Matrixd result(res_rows, res_cols);

        Matrixd temp_propagations = Matrixd::Constant(this->input->get_tensor_height(),
                                                              this->input->get_tensor_width(), 0);

        
        
        Matrixd::Index x_index, y_index;
        for (int instance = 0; instance < this->input->get_batch_size(); instance++) {
            for (int depth = 0; depth < this->input->get_tensor_depth(); depth++) {
                temp_propagations = Matrixd::Constant(this->input->get_tensor_height(),
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


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Dropout>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

//No time to finish it

libdl::layers::Dropout::Dropout(double p){
    this->probability = p;
}

TensorWrapper libdl::layers::Dropout::forward(TensorWrapper& input){
    if(this->input == nullptr)
        this->input = std::make_unique<TensorWrapper>(input);
    *(this->input) = input;


    return *(this->output);
}   

TensorWrapper libdl::layers::Dropout::get_mask(){
    return *(this->mask);
}

TensorWrapper libdl::layers::Dropout::backward(TensorWrapper& gradient, double lr){
    return gradient;
}

void libdl::layers::Dropout::generate_mask(){
    if(this->mask == nullptr)
        this->mask = std::make_unique<TensorWrapper>(this->input->get_batch_size(), this->input->get_tensor_height(),
                this->input->get_tensor_width(), this->input->get_tensor_depth());
    this->mask->get_tensor() = Matrixd::Constant(this->mask->get_tensor().rows(), this->mask->get_tensor().cols(), 0);

    this->mask->get_tensor() = this->mask->get_tensor().unaryExpr([this](double e){
        std::random_device rd;
        std::mt19937 gen(rd());

        std::bernoulli_distribution dist(this->probability);

        if(dist(gen) == true)
            return 1;
        else
            return 0;
    }).cast<double>();
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Dropout>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////