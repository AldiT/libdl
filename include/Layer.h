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
        virtual Tensor forward(Tensor& input) = 0;

        //backward pass function
        virtual Tensor backward(Tensor& gradient, double lr) = 0;



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


class libdl::layers::DenseLayer2D: public libdl::layers::Layer<Eigen::MatrixXd>{
public:

    DenseLayer2D(int, int, std::string, int);

    Matrixd forward(Matrixd&);
    Matrixd backward(Matrixd& , double );


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
class libdl::layers::Sigmoid : public libdl::layers::Layer<Matrixd>{
public:

    Matrixd forward(Matrixd& input);
    Matrixd backward(Matrixd& gradients, double lr);


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
class libdl::layers::Convolution2D : public libdl::layers::Layer<TensorWrapper>{
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

protected:
    //protected because later I might want to implement some fancy convolution layer to perform segmantation or whatever
    //methods

    //Correlation should be the same as convolution in the case of NN so that is what I implement here
    // for simplicity
    TensorWrapper filter_conv(TensorWrapper& gradients_, TensorWrapper&);
    TensorWrapper input_conv (TensorWrapper& gradients_);
    

    
    
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

    Matrixd forward(TensorWrapper& input);
    TensorWrapper backward(Matrixd& gradients);

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


class libdl::layers::Softmax : public libdl::layers::Layer<Matrixd>{
public:

    Matrixd forward(Matrixd&);
    Matrixd backward(Matrixd&, double);

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


class libdl::layers::ReLU : public libdl::layers::Layer<Matrixd>
{
public:

    Matrixd forward(Matrixd& input){

        if(this->input == nullptr)
            this->input = std::make_unique<Matrixd>(input);

        *(this->input) = input;

        auto output = input.unaryExpr([](double e){return ((e > 0)? e : 0.001*e);});

        //std::cout << "Output relu:\n" << input.row(0) << std::endl;

        return output;
    }
    Matrixd backward(Matrixd& gradients, double lr){

        gradients = gradients.array() * this->input->unaryExpr([](double e){return (e > 0 ? 1 : 0.001);}).array();

        return gradients;
    }

private:
    std::unique_ptr<Matrixd> input;

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


#endif //LIBDL_LAYERS_LAYER_H
