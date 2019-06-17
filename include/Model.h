//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_MODEL_H
#define LIBDL_MODEL_H

#include <iostream>
#include <string>
#include "Layer.h"
#include "TensorWrapper.h"

#include <memory>
#include <list>

namespace libdl::model{
    template <typename TensorType>
    class Model;
    class History;
    struct Milestone;
    class Optimizer; //TODO: Implement this in a seperate class
}

typedef libdl::model::Milestone milestone;




struct libdl::model::Milestone{
    std::string name;
    std::string summary;
    std::string value;
};

class libdl::model::History{
public:
    History();

    std::string getMessage();
    void setMessage(std::string);

    std::list<milestone> getHistory();
    bool addHistory(milestone);

    void clearHistory();

protected:

private:
    std::string msg;
    std::list<milestone> history;
};



///
//// This class should build a model based on layers provided by the Layer header file
//// Functionality should include things like training, testing, summary etc.
///
template <typename TensorType>
class libdl::model::Model {
public:
    Model() {};


    void add(libdl::layers::Layer<TensorType> layer);


    libdl::model::History train(libdl::TensorWrapper_Exp&, int epochs, double lr, double lr_decay, int batch_size);

    libdl::model::History test();


protected:


private:
    bool train_mode; //train or test mode
    int training_epochs;
    double learning_rate;
    double learning_rate_decay;
    int batch_size;
    //std::unique_ptr<libdl::model::Optimizer> optimizer; //Incomplete type
    std::list<libdl::layers::Layer<TensorType>> model;
    std::unique_ptr<libdl::model::History> history;
};




#endif //LIBDL_MODEL_H
