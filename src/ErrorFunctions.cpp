//
// Created by Aldi Topalli on 2019-05-19.
//
#include <memory>
#include <iostream>
#include "ErrorFunctions.h"
#include "Eigen/Dense"
#include <cmath>

using namespace libdl::error;

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <ErrorFunctions>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::error::ErrorFunctions::ErrorFunctions(int num_classes, Vectord targets): num_classes(num_classes){
    this->targets = std::make_unique<Vectord>(targets);
}


double libdl::error::ErrorFunctions::get_error(Vectord targets, Vectord logits) {

    if (logits.rows() != this->targets->rows()){
        std::cerr << "Targets number not the same as logits. " << logits.rows() << " !=  " << this->targets->rows()
        << std::endl;
        std::exit(-1);
    }


    this->logits = std::make_unique<Vectord>(logits);

    return (*(this->targets) - *(this->logits)).unaryExpr([](double e){ return std::pow(e, 2);}).sum()/2;
}

Vectord libdl::error::ErrorFunctions::get_gradient() {
    return (*(this->logits) - *(this->targets));
}


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </ErrorFunctions>                         /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <CrossEntropy>                            /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


libdl::error::CrossEntropy::CrossEntropy(int num_classes) {
    this->num_classes = num_classes;
    this->errors = std::vector<double>();
}

double libdl::error::CrossEntropy::get_error(Vectord targets, Matrixd logits) {
    try{

        if(this->logits == nullptr)
            this->logits = std::make_unique<Matrixd>(logits);
        else
            *(this->logits) = logits;

        if(this->targets == nullptr)
            this->targets = std::make_unique<Vectord>(targets);
        else
            *(this->targets) = targets;


        if(targets.rows() != logits.rows())//num of instances should be the same
        {
            throw std::invalid_argument("CrossEntropy::get_error: number of instances does not match!");
        }

        std::cout << "Logits before error: " << *(this->logits) << std::endl;
        
        Matrixd m = *(this->logits);
        //(*(this->logits))(i, targets(i)).dot(Matrixd::Constant(1, 1, 1)) 
        double res = 0;
        for (int i = 0 ; i < this->logits->rows(); i++){
            res += std::log(m(i, targets(i)));
            std::cout << "Log of this: " << (*(this->logits))(i, targets(i)) << std::endl;
        }


        //sum over the classes

        return -res;
    }catch(std::invalid_argument &err){
        std::cerr << "Invalid argument: " << err.what() << std::endl;
        std::exit(-1);
    }catch(std::bad_alloc &err){
        std::cerr << "Bad alloc: " << err.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }
}

Vectord libdl::error::CrossEntropy::predictions(Matrixd logits, Vectord targets) {
    try {
        if (this->logits == nullptr)
            this->logits = std::make_unique<Matrixd>(logits);

        *(this->logits) = logits;

        double acc = 0;

        if (this->targets == nullptr)
            this->targets = std::make_unique<Vectord>(targets);

        *(this->targets) = targets;

        if (targets.rows() != logits.rows())//num of instances should be the same
        {
            std::cout << "Target: " << targets.rows() << " logits: " << logits.rows() << std::endl;
            throw std::invalid_argument("CrossEntropy::get_error: number of instances does not match!");
        }

        Vectord predictions(this->targets->rows());
        Vectord predicted_class(this->targets->rows());

        predictions = this->softmax();
        int index;

        for (int i = 0; i < this->targets->rows(); i++) {
            predictions(i) = this->logits->row(i).maxCoeff(&index);
            predicted_class(i) = index;

            if(predicted_class(i) == targets(i))
                acc += 1;
        }

        //printing
        std::cout << "Prediction results: \n";
        for(int i = 0; i < this->targets->rows(); i++){
            std::cout << "Prediction: " << predicted_class(i) << " Logit: " << predictions(i) << " Label: " << (*(this->targets))(i) << std::endl;
        }

        std::cout << "Test accuracy: " << acc/targets.rows() << std::endl;


        return predictions;
    }catch (std::invalid_argument &err){
        std::cerr << "Invalid argument: " << err.what() << std::endl;
        std::exit(-1);
    }catch(std::bad_alloc &err){
        std::cerr << "Bad alloc: " << err.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }

}


Matrixd libdl::error::CrossEntropy::get_gradient(Matrixd logits, Vectord targets , int iteration) {
    try{

        if (this->logits == nullptr)
            this->logits = std::make_unique<Matrixd>(logits);

        *(this->logits) = logits;

        //std::cout << "Logits before any change: " << logits << std::endl;

        if (this->targets == nullptr)
            this->targets = std::make_unique<Vectord>(targets);

        *(this->targets) = targets;

        Vectord predictions = this->softmax();


        for (int i = 0 ; i < this->logits->rows(); i++){
            avg += -std::log((*(this->logits))(i, targets(i)).dot(Matrixd::Constant(1, 1, 1)));
        }




        Matrixd gradients(this->logits->rows(), this->logits->cols());
        gradients = *(this->logits);



        /*
        for(int row = 0; row < gradients.rows(); row++){
            gradients.row(row) /= sums(row);
        }*/

        for(int instance = 0; instance < this->logits->rows(); instance ++){
            int index = (*(this->targets))(instance);
            gradients(instance, index) -= 1; //loss gradient

        }



        gradients = gradients.unaryExpr([this](double e){
            if(e > 1){
                e = .5;
            }else if(e < -1){
                e = -.5;
            }

            return e / this->logits->rows();
        });//Normalization

        if(iteration % 20 == 0 && iteration != 0) {
            //std::cout << "\nIteration: " << iteration << std::endl;
            std::cout << "[Error: " << avg / (targets.rows()*20) << "; Epoch: " << iteration/20 << "]\n";
            avg = 0;
            //std::cout << "Gradients:\n" << gradients << std::endl;
            //std::cout << "Logits: \n" << *(this->logits) << std::endl;
        }

        return gradients;

    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }
}


Vectord CrossEntropy::softmax() {
    Vectord maximums = this->logits->rowwise().maxCoeff();
    

    this->logits->colwise() -= maximums;

    Vectord sums = this->logits->unaryExpr([](double e){return std::exp(e);}).rowwise().sum();
    //double sum = this->logits->block(instance, 0, 1, this->logits->cols()).sum();

    *(this->logits) = this->logits->unaryExpr([](double e){ return std::exp(e);});

    for(int i = 0; i < this->logits->rows(); i++){
        this->logits->row(i) /= sums(i);
    }

    if(this->logits->minCoeff() < 0)
        std::cout << "LOGIT SMALLER THAN 0 STILL SOME PROBLEM HERE.\n";

    return this->logits->rowwise().maxCoeff();
}



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </CrossEntropy>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////