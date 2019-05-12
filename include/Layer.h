//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_LAYERS_LAYER_H
#define LIBDL_LAYERS_LAYER_H


#include "Eigen/Dense"
#include <string>

namespace libdl::layers {
    class Layer;
    class DenseLayer;
    class Perceptron;
    class Sigmoid;
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

class libdl::layers::Layer {
    public:
        Layer();
        Layer(int){};
        ~Layer();

        //forward pass function
        virtual Eigen::MatrixXd forward(Eigen::MatrixXd input) = 0;

        //backward pass function
        virtual Eigen::MatrixXd backward(Eigen::MatrixXd gradient) = 0;

        virtual int printCrap() = 0;


    private:


    protected:
        int num_neurons;
        Eigen::MatrixXd weights;
        Eigen::MatrixXd biases;
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


class libdl::layers::DenseLayer: protected libdl::layers::Layer{
public:
    DenseLayer(int num_neurons);

    Eigen::MatrixXd forward(Eigen::MatrixXd input);
    Eigen::MatrixXd backward(Eigen::MatrixXd gradient);

    int printCrap(){
        std::cout << "This should get printed from the test cases!" << std::endl;
        return 0;
    }

    int rows(){
        return this->weights_to_neurons.rows();
    }

protected:
    Eigen::MatrixXd weights_to_neurons;
};

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </DenseLayer>                             /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Perceptron>                              /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


// TODO: Add the functionality of the forward pass
class libdl::layers::Perceptron: libdl::layers::Layer{
public:

    Eigen::MatrixXd forward(Eigen::MatrixXd input);
    Eigen::MatrixXd backward(Eigen::MatrixXd gradient);

protected:

private:
};

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Perceptron>                             /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Sigmoid>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

// IDEA: Maybe create a new namespace : activations
class libdl::layers::Sigmoid : libdl::layers::Layer{
public:

    Eigen::MatrixXd forward(Eigen::MatrixXd input);
    Eigen::MatrixXd backward(Eigen::MatrixXd gradients);

protected:

private:
    double sigmoid(double input);
    //There are no weights nor biases, so is it worth it to keep it on this hierarchy structure?
};



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////s

#endif //LIBDL_LAYERS_LAYER_H
