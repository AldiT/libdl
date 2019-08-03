//
// Created by Aldi Topalli on 2019-05-07.
//

#ifndef LIBDL_MODEL_H
#define LIBDL_MODEL_H

#include <iostream>
#include <string>
#include "Layer.h"
#include "TensorWrapper.h"
#include "ErrorFunctions.h"

#include <memory>
#include <list>
#include <vector>

namespace libdl::model{
    class Model;
    class History;
    struct Milestone;
    class Optimizer; //TODO: Implement this in a seperate class
}

typedef libdl::model::Milestone     milestone;
typedef libdl::layers::Layer        Layer;
typedef libdl::TensorWrapper_Exp    TensorWrapper;
typedef Eigen::MatrixXd             Matrixd;
typedef double                      scalar;
typedef int                         whole_number;



struct libdl::model::Milestone{
    std::string name;
    std::string summary;
    scalar value;
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

class libdl::model::Optimizer{
public:

protected:

private:

};



///
//// This class should build a model based on layers provided by the Layer header file
//// Functionality should include things like training, testing, summary etc.
///
class libdl::model::Model {
public:
    Model(whole_number epochs_, scalar lr,scalar lr_decay_, 
    whole_number batch_size_, whole_number num_batches_, std::string optimizer_, 
    std::string loss_function, whole_number num_classes_);

    
    void add(Layer *layer);

    TensorWrapper forward(TensorWrapper&);

    TensorWrapper backward(TensorWrapper& logits, TensorWrapper targets);

    void set_lr(scalar new_lr);


protected:


private:
    bool train_mode; //train or test mode
    whole_number epochs;
    scalar learning_rate;
    scalar lr_decay;
    whole_number batch_size;
    whole_number num_batches;
    whole_number num_classes;

    //std::unique_ptr<libdl::model::Optimizer> optimizer; //Incomplete type
    std::list<libdl::layers::Layer*> layers;
    std::unique_ptr<libdl::model::Optimizer> optimizer;
    std::unique_ptr<libdl::error::CrossEntropy> error;
};




#endif //LIBDL_MODEL_H
