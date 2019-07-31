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
    this->history = std::list<milestone>();

}

std::string libdl::model::History::getMessage() {
    return this->msg;
}

void libdl::model::History::setMessage(std::string msg_) {
    this->msg = msg_;
}

std::list<milestone> libdl::model::History::getHistory() {
    return this->history;
}

bool libdl::model::History::addHistory(milestone to_add) {
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



void libdl::model::Model::add(libdl::layers::Layer *layer_, std::string activation_) {
    try{
        
    }catch(std::invalid_argument &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

libdl::model::History libdl::model::Model::train(TensorWrapper_Exp& train_data_, int epochs_,
        double lr_, double lr_decay_, int batch_size_, std::string optimizer_) {
    
    try{
        *(this->train_data) = train_data_;
        this->epochs = epochs_;
        this->batch_size = batch_size_;
        this->learning_rate = lr_;
        this->lr_decay = lr_decay_;
        this->train_mode = true;

        


    }catch(std::exception &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

libdl::model::History libdl::model::Model::test() {

}

////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Model>                                  /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

