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
    this->avg = std::make_unique<Matrixd>(1, 1);
    (*(this->avg))(0, 0) = 0;
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
        /* 
        double res = 0;
        for (int i = 0 ; i < this->logits->rows(); i++){
            res += std::log(m(i, targets(i)));
            std::cout << "Log of this: " << (*(this->logits))(i, targets(i)) << std::endl;
        }*/


        //sum over the classes

        return 0;//-res;
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

        Vectord max_logits(this->targets->rows());
        Vectord predicted_class(this->targets->rows());

        Matrixd predictions = this->softmax();
        int index;


        for (int i = 0; i < this->targets->rows(); i++) {
            max_logits(i) = predictions.row(i).maxCoeff(&index);
            predicted_class(i) = index;

            if(predicted_class(i) == targets(i))
                acc += 1;
        }

        //printing
        std::cout << "Prediction results: \n";
        for(int i = 0; i < this->targets->rows(); i++){
            std::cout << "Prediction: " << predicted_class(i) << " Logit: " << max_logits(i) << " Label: " << (*(this->targets))(i) << std::endl;
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
        /* 
        std::cout << "\n\n";
        std::cout << "Stats about logits\n";
        std::cout << "Max logit: " << this->logits->maxCoeff() << std::endl;
        std::cout << "Min logit: " << this->logits->minCoeff() << std::endl;
        std::cout << "Avg logit: " << this->logits->mean() << std::endl;
        std::cout << "End of stats about logits\n";
        */
        //std::cout << "Logits before any change: " << logits << std::endl;

        if (this->targets == nullptr)
            this->targets = std::make_unique<Vectord>(targets);

        *(this->targets) = targets;
        
        
        Matrixd gradients(this->logits->rows(), this->logits->cols());
        gradients = this->softmax();
        Matrixd predictions = gradients;
        /* 
        std::cout << "\n";
        std::cout << "Stats about gradient after softmax\n";
        std::cout << "Max logit: " << gradients.maxCoeff() << std::endl;
        std::cout << "Min logit: " << gradients.minCoeff() << std::endl;
        std::cout << "Avg logit: " << gradients.mean() << std::endl;
        std::cout << "End of stats about logits\n";
*/
        Matrixd error_vector(gradients.rows(), 1);
        for (int row = 0; row < error_vector.rows(); row++){
            error_vector.block(row, 0, 1, 1) = gradients.block(row, targets(row), 1, 1);
        }
        


        error_vector = error_vector.unaryExpr([](double e){return -std::log(e);});
        //gradients = gradients.unaryExpr([](double e){return -std::log(e);});
        
        (*(this->avg))(0, 0) += error_vector.sum();
        
        for(int row = 0; row < gradients.rows(); row++){
           gradients.block(row, targets(row), 1, 1) = gradients.block(row, targets(row), 1, 1).unaryExpr([](double e){return e-1;});
        }

    
        /* 
        std::cout << "\n";
        std::cout << "Stats about gradient before return\n";
        std::cout << "Max logit: " << gradients.maxCoeff() << std::endl;
        std::cout << "Min logit: " << gradients.minCoeff() << std::endl;
        std::cout << "Avg logit: " << gradients.mean() << std::endl;
        std::cout << "End of stats about logits\n";
*/
        /*
        for(int row = 0; row < gradients.rows(); row++){
            gradients.row(row) /= sums(row);
        }*/
        /* 
        for(int instance = 0; instance < this->logits->rows(); instance ++){
            int index = (*(this->targets))(instance);
            gradients(instance, index) -= 1; //loss gradient

        }*/


        
        gradients = gradients.unaryExpr([this](double e){
            if(e > 1){
                e = 1;
            }else if(e < -1){
                e = -1;
            }

            return e;
        });//Normalization
        
        int rows = gradients.rows();
        if(iteration % 20 == 0 && iteration != 0) {
            //std::cout << "\nIteration: " << iteration << std::endl;
            std::cout << "[Error: "  << (*(this->avg))(0, 0) / rows << "; Epoch: " << iteration/20 << "]\n";
            (*(this->avg))(0, 0) = 0;
            //std::cout << "Gradients:\n" << gradients << std::endl;
            //std::cout << "Logits: \n" << predictions << std::endl;
        }
        

        return gradients;

    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }
}

Matrixd libdl::error::CrossEntropy::gradient(Matrixd logits, Vectord targets, int epoch, std::string &msg, double& error){
    try{
        if (this->logits == nullptr)
            this->logits = std::make_unique<Matrixd>(logits);

        *(this->logits) = logits;

        if (this->targets == nullptr)
            this->targets = std::make_unique<Vectord>(targets);
        *(this->targets) = targets;

        Vectord predictions = this->softmax();
        
        *(this->logits) = this->logits->unaryExpr([](double e){return std::log(e);});

        for (int i = 0 ; i < this->logits->rows(); i++){
            (*(this->avg))(0, 0) -= this->logits->block(i, targets(i), 1, 1).sum();
        }

        this->batch_size = this->logits->rows();

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

        msg = "[Error: " + std::to_string((*(this->avg))(0, 0) / (this->batch_size)) + "; Epoch: " + std::to_string(epoch) + "]\n";
        error = (*(this->avg))(0, 0);
        (*(this->avg))(0, 0) = 0;
        //std::cout << "Gradients:\n" << gradients << std::endl;
        //std::cout << "Logits: \n" << *(this->logits) << std::endl;

        return -gradients;


    }catch(std::exception &err){
        std::cerr << "Unexcpected error: " << err.what() << std::endl;
        std::exit(-1);
    }
}

//Basically I use this to apply softmax to the logits.
//And it returns the max coefficients from each row (predictions).
Matrixd CrossEntropy::softmax() {

    Matrixd copy_logits = *(this->logits);

    copy_logits.colwise() -= copy_logits.rowwise().maxCoeff();

    copy_logits = copy_logits.unaryExpr([](double e){return std::exp(e);});

    Vectord sums = copy_logits.rowwise().sum();

    for(int i = 0; i < copy_logits.rows(); i++){
        copy_logits.row(i) /= sums(i);
    }
    

    if(copy_logits.minCoeff() < 0)
        std::cout << "LOGIT SMALLER THAN 0 STILL SOME PROBLEM HERE.\n";

    return copy_logits;
}



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </CrossEntropy>                           /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////