//
// Created by Aldi Topalli on 2019-05-07.
//

#include <iostream>

#include "Layer.h"
#include "Eigen/Dense"
#include "spdlog/spdlog.h"

using namespace libdl::layers;

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Layer>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
template <typename Tensor>
Layer<Tensor>::Layer() {

}
template <typename Tensor>
Layer<Tensor>::~Layer() {

}

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


DenseLayer2D::DenseLayer2D(int input_features, int num_neurons, std::string name="Layer"): Layer<Eigen::MatrixXd>::Layer(){
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

Eigen::MatrixXd DenseLayer2D::forward(Eigen::MatrixXd input) {

    try{

        if(this->weights->rows() != input.cols()) {
            std::string msg;
            msg = "Not compatible shapes: " + std::to_string(this->weights->rows()) + " != " +
                    std::to_string(input.cols()) + " !";
            throw msg;
        }

        auto temp = input * *(this->weights);

        //TODO: Add the bias to the weight with input multiplication
        /*
        for(int i = 0; i < input.rows(); i++){
            temp.row(i) = temp.row(i) + *(this->biases);
        }*/


        return temp;

    }catch (const std::string msg){
        std::cerr << msg <<std::endl;
        std::exit(-1);
    }catch(...){
        std::cerr << "Unexpected error happend in the forward pass of layer: " << this->name <<std::endl;
        std::exit(-1);
    }
}

Eigen::MatrixXd DenseLayer2D::backward(Eigen::MatrixXd gradient) {
    return Eigen::MatrixXd::Random(3, 3);
}


std::string DenseLayer2D::info(){
    std::string str;

    str = "Some dummy string";
    return str;
}

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


//TODO: Add the functionality for Perceptron layer
//      namely, define only the forward pass for now!
Eigen::MatrixXd Perceptron::forward(Eigen::MatrixXd input){
    //Eigen::MatrixXd temp = this->weights.dot(input);
    //temp = temp + this->biases;
    return Eigen::MatrixXd::Constant(3, 3, 1);
}

Eigen::MatrixXd Perceptron::backward(Eigen::MatrixXd gradient){
    //TODO: Implement the backward pass for this function
    return Eigen::MatrixXd::Constant(3, 3, 1.2);
}


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

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd input){
    return Eigen::MatrixXd::Constant(3, 3, 1);
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd gradients){
    return Eigen::MatrixXd::Constant(3, 3, 1);
}

double Sigmoid::sigmoid(double input){
    return 2;
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

