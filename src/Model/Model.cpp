//
// Created by Aldi Topalli on 2019-05-07.
//

#include "Model.h"

#include <iostream>
#include <string>
#include <memory>
#include <list>
#include "Layer.h"
#include <functional>


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

libdl::model::Model::Model(whole_number epochs_, scalar lr, scalar lr_decay_, 
    whole_number batch_size_, whole_number num_batches_, std::string optimizer_, 
    std::string loss_function, whole_number num_classes_){
    
    this->epochs = epochs_;
    this->lr_decay = lr_decay_;
    this->batch_size = batch_size_;
    this->num_batches = num_batches_;
    this->num_classes = num_classes_;
    
    if(optimizer_ == "adam")//not working for now
        this->optimizer = std::make_unique<libdl::model::Optimizer>();
    else
        this->optimizer = nullptr;

    if(loss_function == "cross_entropy")
        this->error = std::make_unique<libdl::error::CrossEntropy>(this->num_classes);    

}



void libdl::model::Model::add(Layer *layer_) {
    try{
        this->layers.push_back(layer_);

    }catch(std::bad_alloc &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }catch(std::invalid_argument &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

TensorWrapper libdl::model::Model::forward(TensorWrapper& data_) {
    
    try{
        TensorWrapper out(1, 1, 1, 1);
        TensorWrapper grads(1, 1, 1, 1);

        for(std::list<Layer*>::iterator it = this->layers.begin();
            it != this->layers.end(); ++it){
            
            if(it == this->layers.begin())
                out = (*it)->forward(data_);
            else
                out = (*it)->forward(out);
        }

        
        return out;
    }catch(std::exception &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

TensorWrapper libdl::model::Model::backward(TensorWrapper& logits, TensorWrapper targets) {
    try{
        this->learning_rate = 0.000001;
        TensorWrapper grads(logits.get_batch_size(), logits.get_tensor_height(), 
            logits.get_tensor_width(), logits.get_tensor_depth());
            
        grads.get_tensor() = this->error->get_gradient(logits.get_tensor(), targets.get_tensor(), 20);

        //std::cout << "Incoming gradient:\n " << grads.get_tensor() << std::endl;

        TensorWrapper out(1, 1, 1, 1);

        for(std::list<Layer*>::reverse_iterator it = this->layers.rbegin();
            it != this->layers.rend(); ++it){
            //std::cout << "Some weights before: " << (*it)->get_weights().block(0, 0, 1, 10) << std::endl;
            
            if(it == this->layers.rbegin())
                out = (*it)->backward(grads, this->learning_rate);
            else
                out = (*it)->backward(out, this->learning_rate);

            //std::cout << "Same weights after: " << (*it)->get_weights().block(0, 0, 1, 10) << std::endl;

        }

        
        return out;
    }catch(std::exception &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

void libdl::model::Model::set_lr(scalar factor){
    this->learning_rate *= factor;
}
////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                            </Model>                                  /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////

