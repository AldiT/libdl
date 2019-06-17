//
// Created by Aldi Topalli on 2019-05-07.
//

#include "Model.h"

#include <iostream>
#include <string>
#include <memory>
#include <list>
#include "Layer.h"



////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <History>                                 /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

libdl::model::History::History() {
    this->history = std::list<Milestone>();

}

std::string libdl::model::History::getMessage() {
    return this->msg;
}

void libdl::model::History::setMessage(std::string msg_) {
    this->msg = msg_;
}

std::list<Milestone> libdl::model::History::getHistory() {
    return this->history;
}

bool libdl::model::History::addHistory(Milestone to_add) {
    try{
        this->history.push_back(to_add);
        return true;
    }catch (std::bad_alloc & obj){
        std::cerr << "Bad allocation, most probably not enough memory: " << obj.what() << std::endl;
        return false;
    }
}

void libdl::model::History::clearHistory() {
    this->history.clear();
}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </History>                                /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            <Model>                                   /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

template <typename TensorType>
libdl::model::Model<TensorType>::Model() {
}

//Model
template <typename TensorType>
void libdl::model::Model<TensorType>::add(libdl::layers::Layer<TensorType>* layer_) {
    this->model.push_back(std::make_unique<libdl::TensorWrapper_Exp>(layer_));
}

template <typename TensorType>
libdl::model::History libdl::model::Model<TensorType>::train(TensorWrapper_Exp& train_data, int epochs, double lr,
        double lr_decay, int batch_size)
{
    /*
    this->training_epochs = epochs;
    this->learning_rate = lr;
    this->learning_rate_decay = lr_decay;
    this->batch_size = batch_size;
    *(this->optimizer) = optimizer;

    for(int epoch = 0; epoch < this->training_epochs; epoch++){
        for(int batch = 0; batch < train_data.size()%this->batch_size; batch++){
            //TODO : This is the main loop that should do the forward and backward pass for the given training data

            //FORWARD PASS--> TODO: It would be a better idea to put this in a seperate private function
            for(std::iterator it = this->model->begin(); it != this->model->end(); it++){

            }

            //BACKWARD PASS-->TODO: Also put this in a seperate function and just call here two functions
            for(;false;){

            }
        }
    }*/

    //TODO: Needs to return what it needs to return sports
}

template <typename TensorType>
libdl::model::History libdl::model::Model<TensorType>::test() {

}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Model>                                  /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

