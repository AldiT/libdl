//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_LAYERS_LAYER_H
#define LIBDL_LAYERS_LAYER_H

#include <memory>
#include <vector>
#include <string>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "TensorWrapper.h"
#include <pybind11/pybind11.h>
#include <math.h>

namespace libdl::layers {
    class Layer;
    class PyLayer;
    class DenseLayer2D;
    class Perceptron;
    class Sigmoid;
    //template <typename Tensor>
    class Convolution2D;
    class Flatten;
    class Softmax;
    class ReLU;
    class MaxPool;
    class Dropout;
    class TanH;
}


typedef libdl::TensorWrapper_Exp TensorWrapper;
typedef Eigen::MatrixXd Matrixd;

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

class libdl::layers::Layer {
    public:
        Layer() {};
        Layer(int){};
        virtual ~Layer() {};

        //forward pass function
        virtual TensorWrapper forward(TensorWrapper& input){return input;}

        //backward pass function
        virtual TensorWrapper backward(TensorWrapper& gradient, double lr){return gradient;}



    private:


    protected:
        int num_neurons;
        std::unique_ptr<TensorWrapper> weights;
        std::unique_ptr<Eigen::VectorXd> biases;
        std::unique_ptr<TensorWrapper> input;
        std::unique_ptr<TensorWrapper> output;
        std::string name;

};


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Layer>                                  /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <PyLayer>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
/*
class libdl::layers::PyLayer : public libdl::layers::Layer{
public:
    using libdl::layers::Layer::Layer;

    //forward pass function
        TensorWrapper forward(TensorWrapper& input) override {
            PYBIND11_OVERLOAD_PURE(
                TensorWrapper,
                Layer,      
                forward,      
                input    
            );
        };

        //backward pass function
        TensorWrapper backward(TensorWrapper& gradient, double lr) override {
            PYBIND11_OVERLOAD_PURE(
                TensorWrapper, 
                Layer,   
                backward,    
                gradient,
                lr 
            );
        };
};
 */

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </PyLayer>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <DenseLayer>                              /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


class libdl::layers::DenseLayer2D: public libdl::layers::Layer{
public:

    DenseLayer2D(int, int, std::string, int);

    TensorWrapper forward(TensorWrapper&);
    TensorWrapper backward(TensorWrapper& , double );


    int rows(){
        return this->weights->get_tensor().rows();
    }

    std::string info();

    Matrixd get_weights(){
        return this->weights->get_tensor();
    }

    Matrixd get_biases(){
        return *(this->biases);
    }

    std::string get_name(){
        return this->name;
    }


protected:
};

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </DenseLayer>                             /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////








//TODO: Add sigmoid on top of the two Dense Layers you just created.

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Sigmoid>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

// IDEA: Maybe create a new namespace : activations
class libdl::layers::Sigmoid : public libdl::layers::Layer{
public:

    TensorWrapper forward(TensorWrapper& input);
    TensorWrapper backward(TensorWrapper& gradients, double lr);


protected:

private:
    double sigmoid(double input);
    //There are no weights nor biases, so is it worth it to keep it on this hierarchy structure?
};



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


//Still Experimental
////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Convolution2D>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

//template <typename Tensor>
class libdl::layers::Convolution2D : public libdl::layers::Layer{
public:
    //constructor
    Convolution2D(std::string, int kernel_size_=3, int num_filters_=16, int padding_=0, int stride_=1,
            int input_depth_=3, int input_neurons_=144);

    Convolution2D(Matrixd filter);



    TensorWrapper forward(TensorWrapper& input_);
    TensorWrapper backward(TensorWrapper& gradients_, double lr);

    Matrixd& get_filters(){
        return this->filters->get_tensor();
    }
    void set_filters(TensorWrapper& new_filters){
        *(this->filters) = new_filters;
    }
    
    TensorWrapper& pad(TensorWrapper&);
    //rename to dilation
    TensorWrapper& dilation(TensorWrapper&);
    TensorWrapper& convolution_operation() const;
    TensorWrapper reverse_tensor(TensorWrapper&) const;
    TensorWrapper& clean_gradient(TensorWrapper&);

    bool detect_illegal_combination() const;

    //This method is for testing the gradient's correctness
    TensorWrapper get_filter_gradients(){
        return *(this->filter_grad);
    }
    TensorWrapper filter_conv(TensorWrapper gradients_, TensorWrapper&);
    TensorWrapper input_conv (TensorWrapper gradients_);
    
protected:
    //protected because later I might want to implement some fancy convolution layer to perform segmantation or whatever
    //methods

    //Correlation should be the same as convolution in the case of NN so that is what I implement here
    // for simplicity
    

    
    
private:
    std::shared_ptr<TensorWrapper> output;
    std::unique_ptr<TensorWrapper> filters; //Shared because it will point to the same address as weights from Layer
                                     // to save memory
    //biases inherited from Layer
    std::string name;
    int num_filters;
    int kernel_size;
    int input_depth;
    int stride;
    int padding;
    int filter_rank;

    std::unique_ptr<TensorWrapper> filter_grad;
    std::unique_ptr<TensorWrapper> input_grad;

};

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

class libdl::layers::Flatten : public libdl::layers::Layer
{
public:
    Flatten(int batch_size, int height, int width, int depth);

    TensorWrapper forward(TensorWrapper& input);
    TensorWrapper backward(TensorWrapper& gradients, double lr);

protected:

private:
    std::unique_ptr<TensorWrapper> input;
    std::unique_ptr<TensorWrapper> gradient;
};



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


class libdl::layers::Softmax : public libdl::layers::Layer{
public:

    TensorWrapper forward(TensorWrapper&);
    TensorWrapper backward(TensorWrapper&, double);

protected:

private:

};



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Softmax>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <ReLU>                                    /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


class libdl::layers::ReLU : public libdl::layers::Layer
{
public:

    TensorWrapper forward(TensorWrapper& input){

        if(this->input == nullptr)
            this->input = std::make_unique<TensorWrapper>(input);

        *(this->input) = input;

        //std::cout << "Relu\n";

        TensorWrapper output(input);
        output.get_tensor() = input.get_tensor().unaryExpr([](double e){return ((e > 0)? e : 0.001*e);});

        //std::cout << "Output relu:\n" << input.row(0) << std::endl;

        return output;
    }
    TensorWrapper backward(TensorWrapper& gradients, double lr){
        //std::cout << "Relu begin\n";

        gradients.get_tensor() = gradients.get_tensor().array() * this->input->get_tensor().unaryExpr([](double e){return (e > 0 ? 1 : 0.001);}).array();
        
        /* std::cout << "res shape: " << res.rows() << "x" << res.cols() << std::endl;
        std::cout << "Grads shape: " << gradients.get_tensor().rows() << "x" << gradients.get_tensor().cols() << std::endl;
        std::cout << "Input tensor shape: " << this->input->get_tensor().rows() << "x" << this->input->get_tensor().cols() << std::endl;
        std::cout << "Relu end\n";*/
        return gradients;
    }

private:
    std::unique_ptr<TensorWrapper> input;

};


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </ReLU>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <MaxPool>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

class libdl::layers::MaxPool: public libdl::layers::Layer
{
public:
    MaxPool(int kernel, int stride);

    TensorWrapper forward(TensorWrapper&);
    TensorWrapper backward(TensorWrapper&, double);

protected:

private:
    std::unique_ptr<TensorWrapper> past_propagation;
    std::unique_ptr<TensorWrapper> output;
    std::unique_ptr<TensorWrapper> backward_gradient;
    int window_size;
    int stride;


    void max_pooling();


};

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

class libdl::layers::Dropout : public libdl::layers::Layer{
public:
    Dropout(double p);

    TensorWrapper forward(TensorWrapper&);
    TensorWrapper backward(TensorWrapper& gradient, double lr);

    void generate_mask();
    TensorWrapper get_mask();

protected:

private:
    std::unique_ptr<TensorWrapper> input;
    std::unique_ptr<TensorWrapper> output;
    std::unique_ptr<TensorWrapper> mask;
    double probability;

};


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Dropout>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <TanH>                                    /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

class libdl::layers::TanH : public libdl::layers::Layer{
public:

    TensorWrapper forward(TensorWrapper& input){
        if(this->input == nullptr){
            this->input = std::make_unique<TensorWrapper>(input);
        }
        *(this->input) = input;
        TensorWrapper result(input.get_batch_size(), input.get_tensor_height(), input.get_tensor_width(), 
        input.get_tensor_depth());

        result.get_tensor() = this->input->get_tensor().unaryExpr([](double e){ return std::tanh(e);});
        return result;
    }

    TensorWrapper backward(TensorWrapper& gradient, double lr){
        gradient.get_tensor() = gradient.get_tensor().array() * this->input->get_tensor().unaryExpr([](double e){ return 1 - std::pow(std::tanh(e), 2);}).array();
        return gradient;
    }

protected:

private:
    std::unique_ptr<TensorWrapper> input;
    

};

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </TanH>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

#endif //LIBDL_LAYERS_LAYER_H
