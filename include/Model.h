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
    class Optimizer; //TODO: Implement this in a seperate file
}


typedef libdl::model::Milestone Milestone;
typedef libdl::model::History History;



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

    std::list<Milestone> getHistory();
    bool addHistory(Milestone);

    void clearHistory();

protected:

private:
    std::string msg;
    std::list<Milestone> history;
};



///
//// This class should build a model based on layers provided by the Layer header file
//// Functionality should include things like training, testing, summary etc.
///
template <typename TensorType>
class libdl::model::Model {
public:
    Model();


    void add(libdl::layers::Layer<TensorType>* layer);


    libdl::model::History train(libdl::TensorWrapper_Exp&, int epochs, double lr, double lr_decay, int batch_size/*,
            libdl::model::Optimizer optimizer*/); //For now Optimizer let it be just null

    libdl::model::History test();


protected:


private:
    bool train_mode; //train or test mode
    int training_epochs;
    double learning_rate;
    double learning_rate_decay;
    int batch_size;
    std::list<std::unique_ptr<libdl::layers::Layer<TensorType>>> model;
    std::unique_ptr<History> history;
};




#endif //LIBDL_MODEL_H
