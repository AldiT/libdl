//
// Created by Aldi Topalli on 2019-05-19.
//

#ifndef LIBDL_ERRORFUNCTIONS_H
#define LIBDL_ERRORFUNCTIONS_H

#include <memory>
#include "Eigen/Dense"

namespace libdl::error{
    class ErrorFunctions;
    class CrossEntropy;
}

typedef Eigen::VectorXd Vectord;
typedef Eigen::MatrixXd Matrixd;

class libdl::error::ErrorFunctions {
public:
    ErrorFunctions(int num_classes, Vectord targets);

    //TODO: Generalize type Eigen::MatrixXd to TensorWrapper
    double get_error(Vectord targets, Vectord logits);
    Vectord get_gradient();
protected:

private:
    int num_classes;
    std::unique_ptr<Vectord> targets;
    std::unique_ptr<Vectord> logits;
};


class libdl::error::CrossEntropy
{
public:
    CrossEntropy(int num_classes, Vectord targets);

    double get_error(Vectord targets, Matrixd logits);
    Matrixd get_gradient();

protected:

private:
    int num_classes;
    std::unique_ptr<Vectord> targets;
    std::unique_ptr<Matrixd> logits;

    Vectord softmax(int);

};

#endif //LIBDL_ERRORFUNCTIONS_H
