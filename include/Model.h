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
#include <vector>

typedef Eigen::MatrixXd  Matrixd;
typedef libdl::TensorWrapper_Exp TensorWrapper;



namespace libdl::model{
    class Model;
    class History;
    struct Milestone;
    class Optimizer; //TODO: Implement this in a seperate file
}

typedef libdl::model::Milestone milestone;
typedef libdl::layers::Layer<Matrixd> LayerM;
typedef libdl::layers::Layer<TensorWrapper> LayerT;

typedef libdl::model::Milestone Milestone;
typedef libdl::model::History History;



struct libdl::model::Milestone{
    std::string name;
    std::string summary;
    double value;
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
class libdl::model::Model {
public:
    Model();


    void add(LayerM *layer, std::string activation_="none");
    void add(LayerT *layer, std::string activation_="none");

    libdl::model::History train(libdl::TensorWrapper_Exp& train_data, int epochs, double lr,
         double lr_decay, int batch_size, std::string optimizer_);

    libdl::model::History test();


protected:


private:
    bool train_mode; //train or test mode
    int epochs;
    double learning_rate;
    double lr_decay;
    int batch_size;
    //std::unique_ptr<libdl::model::Optimizer> optimizer; //Incomplete type

    std::list<LayerM*> dense_layers;
    std::list<LayerT*> complex_layers;
    std::list<LayerM*> activation_layers;
    std::vector<std::string> activations;
    std::unique_ptr<libdl::model::History> history;
    std::unique_ptr<TensorWrapper> train_data;
};




#endif //LIBDL_MODEL_H
