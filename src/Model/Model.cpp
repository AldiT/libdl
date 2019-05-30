//
// Created by Aldi Topalli on 2019-05-07.
//

#include "Model.h"

#include <iostream>
#include <memory>
#include <list>
#include "Layer.h"



//Model
template <typename TensorType>
void libdl::model::Model<TensorType>::add(libdl::layers::Layer<TensorType> layer) {
    //this->model.push_back(layer);
}

template <typename TensorType>
libdl::model::History libdl::model::Model<TensorType>::train(int epochs, double lr, double lr_decay, int batch_size,
                     libdl::model::Optimizer optimizer) {
    History h;
    return  h;
}

template <typename TensorType>
libdl::model::History libdl::model::Model<TensorType>::test() {

}



//History

