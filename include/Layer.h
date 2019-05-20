//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_LAYERS_LAYER_H
#define LIBDL_LAYERS_LAYER_H

#include <memory>
#include "Eigen/Dense"
#include <string>


namespace libdl::layers {
    template <typename Tensor>
    class Layer;
    class DenseLayer2D;
    class Perceptron;
    class Sigmoid;
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

template <typename Tensor>
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
        std::unique_ptr<Tensor> weights = nullptr;
        std::unique_ptr<Eigen::VectorXd> biases = nullptr;
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

    Eigen::MatrixXd forward(Eigen::MatrixXd input);
    Eigen::MatrixXd backward(Eigen::MatrixXd gradient);

    int printCrap(){
        std::cout << "This should get printed from the test cases!" << std::endl;
        return 0;
    }

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
    std::unique_ptr<Eigen::MatrixXd> input;
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

class libdl::layers::Perceptron: libdl::layers::Layer<Eigen::MatrixXd>{
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
    Eigen::MatrixXd backward(Eigen::MatrixXd gradients);

    int printCrap(){
        return 1;
    }

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
