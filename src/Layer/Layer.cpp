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
//template <typename Tensor>
//Layer<Tensor>::Layer() {
//
//}
//template <typename Tensor>
//Layer<Tensor>::~Layer() {
//
//}

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

        this->input = std::make_unique<Eigen::MatrixXd>(input);

        if(this->weights->rows() != input.cols()) {
            std::string msg;
            msg = "Not compatible shapes: " + std::to_string(this->weights->rows()) + " != " +
                    std::to_string(input.cols()) + " !";
            throw msg;
        }

        Eigen::MatrixXd temp;
        temp = input * *(this->weights);

        temp.rowwise() += this->biases->transpose();


        return temp;

    }catch (const std::string msg){
        std::cerr << msg <<std::endl;
        std::exit(-1);
    }catch(...){
        std::cerr << "Unexpected error happend in the forward pass of layer: " << this->name <<std::endl;
        std::exit(-1);
    }
}

Eigen::MatrixXd DenseLayer2D::backward(Eigen::MatrixXd gradient, double lr) {
    //update weights
    *(this->weights) -= lr * (this->input->transpose() * gradient); // replace 4 by N
    *(this->biases) -= lr * gradient.colwise().sum().transpose();

    return gradient * this->weights->transpose();
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

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd input){
    this->input = std::make_unique<Eigen::MatrixXd>(input);

    return input.unaryExpr([this](double e){ return this->sigmoid(e);});
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd gradient, double lr){
    //std::cout << "\nShape of temp:\n" << temp.rows() << "x" << temp.cols() << std::endl;
    //std::cout << "\nShape of gradient: \n" << gradient.rows() << "x" << gradient.cols() << std::endl;

    return this->input->unaryExpr([this](double e)
    {return this->sigmoid(e) * (1 - this->sigmoid(e));}).array() * gradient.array();
}

double Sigmoid::sigmoid(double input){
    return 1 / (1 + std::exp(-input));
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Sigmoid>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

