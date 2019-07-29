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
void libdl::model::Model::add(libdl::layers::Layer<Matrixd> *layer_, std::string activation_) {
    try{
        this->dense_layers.push_back(layer_);

        if(activation_ == "none"){
            this->activations.push_back("none");
        }else if(activation_ == "relu"){
            this->activations.push_back("relu");
            this->activation_layers.push_back(new libdl::layers::ReLU());
        }else if(activation_ == "sigmoid"){
            this->activations.push_back("sigmoid");
            this->activation_layers.push_back(new libdl::layers::Sigmoid());
        }else{
            throw std::invalid_argument("No known activation with this name: " + activation_ + "!");
        }
    }catch(std::invalid_argument &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }catch(std::exception &exp){
        std::cerr << "Model::train: " << exp.what() << std::endl;
        std::exit(-1);
    }
}

void libdl::model::Model::add(libdl::layers::Layer<TensorWrapper> *layer_, std::string activation_) {
    try{
        this->complex_layers.push_back(layer_);

        if(activation_ == "none"){
            this->activations.push_back("none");
        }else if(activation_ == "relu"){
            this->activations.push_back("relu");
            this->activation_layers.push_back(new libdl::layers::ReLU());
        }else if(activation_ == "sigmoid"){
            this->activations.push_back("sigmoid");
            this->activation_layers.push_back(new libdl::layers::Sigmoid());
        }else{
            throw std::invalid_argument("No known activation with this name: " + activation_ + "!");
        }
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

        if(this->complex_layers.empty() && this->dense_layers.empty())
            throw std::invalid_argument("There is no model to train!");

        if(this->dense_layers.empty())
            throw std::invalid_argument("Model has no dense layers, fully convolution network not supported!");

        std::list<LayerM*>::iterator dense_layer_it = this->dense_layers.begin();
        std::list<LayerT*>::iterator complex_layers_it = this->complex_layers.begin();

        int layer = 0;

        for(int epoch = 1; epoch <= this->epochs; epoch++){
            for(int layer = 0; layer < this->activations.size(); layer++){
                
            }
        }


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

