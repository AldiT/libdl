//
// Created by Aldi Topalli on 2019-05-19.
//

#ifndef LIBDL_ERRORFUNCTIONS_H
#define LIBDL_ERRORFUNCTIONS_H

#include <memory>
#include "Eigen/Dense"

namespace libdl::error{
    class ErrorFunctions;
}

class libdl::error::ErrorFunctions {
public:
    ErrorFunctions(int num_classes, Eigen::VectorXd targets);

    //TODO: Generalize type Eigen::MatrixXd to TensorWrapper
    double get_error(Eigen::VectorXd targets, Eigen::MatrixXd logits);
    Eigen::VectorXd get_gradient();
protected:

private:
    int num_classes;
    std::unique_ptr<Eigen::VectorXd> targets;
    std::unique_ptr<Eigen::MatrixXd> logits;
};


#endif //LIBDL_ERRORFUNCTIONS_H
