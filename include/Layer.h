//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_LAYERS_LAYER_H
#define LIBDL_LAYERS_LAYER_H

#include <memory>
#include <vector>
#include <string>
#include "Eigen/Dense"
#include "TensorWrapper.h"

namespace libdl::layers {
    template <typename Tensor>
    class Layer;
    class DenseLayer2D;
    class Perceptron;
    class Sigmoid;
    //template <typename Tensor>
    class Convolution2D;
    class Flatten;
    class Softmax;
    class ReLU;
    class MaxPool;
}


typedef libdl::TensorWrapper_Exp TensorWrapper;
typedef Eigen::MatrixXd Matrixd;

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

template <typename Tensor>
class libdl::layers::Layer {
    public:
        Layer() {};
        Layer(int){};
        ~Layer() {};

        //forward pass function
        virtual Tensor& forward(Tensor& input) = 0;

        //backward pass function
        virtual Tensor& backward(Tensor& gradient, double lr) = 0;



    private:


    protected:
        int num_neurons;
        std::unique_ptr<Tensor> weights;
        std::unique_ptr<Eigen::VectorXd> biases;
        std::unique_ptr<Tensor> input;
        std::unique_ptr<Tensor> output;
        std::string name;

};


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


class libdl::layers::DenseLayer2D: protected libdl::layers::Layer<Eigen::MatrixXd>{
public:

    DenseLayer2D(int, int, std::string);

    Matrixd& forward(Matrixd&);
    Matrixd& backward(Matrixd& , double );


    int rows(){
        return this->weights->rows();
    }

    std::string info();

    Matrixd get_weights(){
        return *(this->weights);
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
class libdl::layers::Sigmoid : libdl::layers::Layer<Matrixd>{
public:

    Matrixd& forward(Matrixd& input);
    Matrixd& backward(Matrixd& gradients, double lr);


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
class libdl::layers::Convolution2D : libdl::layers::Layer<TensorWrapper>{
public:
    //constructor
    Convolution2D(int kernel_size_=3, int num_filters_=16, int stride_=1, int padding_=0, int input_depth_=3);

    Convolution2D(Matrixd filter);



    TensorWrapper& forward(TensorWrapper& input_);
    TensorWrapper& backward(TensorWrapper& gradients_, double lr);


protected:
    //protected because later I might want to implement some fancy convolution layer to perform segmantation or whatever
    //methods

    //Correlation should be the same as convolution in the case of NN so that is what I implement here
    // for simplicity

    Matrixd rotate180(Eigen::MatrixXd filter);



private:
    std::unique_ptr<TensorWrapper> output;
    std::unique_ptr<TensorWrapper> filters; //Shared because it will point to the same address as weights from Layer
                                     // to save memory
    //biases inherited from Layer
    int num_filters;
    int kernel_size;
    int input_depth;
    int stride;
    int padding;
    int filter_rank;


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

class libdl::layers::Flatten
{
public:
    Flatten(int batch_size, int height, int width, int depth);

    Matrixd& forward(TensorWrapper& input);
    TensorWrapper& backward(Matrixd& gradients);

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


class libdl::layers::Softmax : libdl::layers::Layer<Matrixd>{
public:

    Matrixd& forward(Matrixd&);
    Matrixd& backward(Matrixd&, double);

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


class libdl::layers::ReLU
{
public:

    Matrixd& forward(Matrixd& input){
        input.unaryExpr([](double e){return ((e > 0)? e : 0);});
        return input;
    }
    Matrixd& backward(Matrixd& gradients, double lr){
        gradients.unaryExpr([](double e){return (e > 0 ? e : 0);});
        return gradients;
    }

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

class libdl::layers::MaxPool: public libdl::layers::Layer<TensorWrapper>
{
public:
    MaxPool(int kernel, int stride);

    TensorWrapper& forward(TensorWrapper&);
    TensorWrapper& backward(TensorWrapper&, double);

protected:

private:
    std::unique_ptr<TensorWrapper> past_propagation;
    std::unique_ptr<TensorWrapper> output;
    std::unique_ptr<TensorWrapper> backward_gradient;
    int window_size;
    int stride;


    Matrixd max_pooling(Matrixd, Matrixd&);


};

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </MaxPool>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


#endif //LIBDL_LAYERS_LAYER_H
