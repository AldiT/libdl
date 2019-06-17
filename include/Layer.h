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

namespace libdl{}

namespace libdl::layers {
    template <typename Tensor>
    class Layer;
    class DenseLayer2D;
    class Perceptron;
    class Sigmoid;
    //template <typename Tensor>
    class Convolution2D;
}


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
        virtual Tensor forward(Tensor input) = 0;

        //backward pass function
        virtual Tensor backward(Tensor gradient, double lr) = 0;



    private:


    protected:
        int num_neurons;
        std::unique_ptr<Tensor> weights;
        std::unique_ptr<Eigen::VectorXd> biases;
        std::unique_ptr<Tensor> input;
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

    Eigen::MatrixXd forward(Eigen::MatrixXd );
    Eigen::MatrixXd backward(Eigen::MatrixXd , double );


    int rows(){
        return this->weights->rows();
    }

    std::string info();

    Eigen::MatrixXd get_weights(){
        return *(this->weights);
    }

    Eigen::MatrixXd get_biases(){
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
class libdl::layers::Sigmoid : libdl::layers::Layer<Eigen::MatrixXd>{
public:

    Eigen::MatrixXd forward(Eigen::MatrixXd input);
    Eigen::MatrixXd backward(Eigen::MatrixXd gradients, double lr);


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
class libdl::layers::Convolution2D : libdl::layers::Layer<libdl::TensorWrapper_Exp>{
public:
    //constructor
    Convolution2D(int kernel_size_=3, int num_filters_=16, int stride_=1, int padding_=0, int input_depth_=3);

    Convolution2D(Eigen::MatrixXd filter);



    libdl::TensorWrapper_Exp forward(libdl::TensorWrapper_Exp input_);
    libdl::TensorWrapper_Exp backward(libdl::TensorWrapper_Exp gradients_, double lr);


protected:
    //protected because later I might want to implement some fancy convolution layer to perform segmantation or whatever
    //methods

    //Correlation should be the same as convolution in the case of NN so that is what I implement here
    // for simplicity

    Eigen::MatrixXd rotate180(Eigen::MatrixXd filter);



private:
    std::unique_ptr<libdl::TensorWrapper_Exp> output;
    std::unique_ptr<libdl::TensorWrapper_Exp> filters; //Shared because it will point to the same address as weights from Layer
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

#endif //LIBDL_LAYERS_LAYER_H
