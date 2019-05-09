//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSORWRAPPER_H
#define LIBDL_TENSORWRAPPER_H

#include "Eigen/Dense"



namespace libdl::tensor{
    template <typename scalarType>
    class TensorWrapper;
}


template <typename scalarType >
class libdl::tensor::TensorWrapper {

public:
    //Add a constructor that takes an arbitrary number of values , which represent an arbitrary number of
    // dimensions.
    TensorWrapper(){};
    TensorWrapper(int* dims);
    TensorWrapper(TensorWrapper* tensor);


    //Also overload operators for multiplication, indexing and addition
    TensorWrapper operator() (int* indexes);//Get a specific value in the tensor
    TensorWrapper operator* (TensorWrapper* tensor);//Multiply two tensors with compatible sizes
    TensorWrapper operator+ (scalarType scalar);//Multiply each element of the tensor with a scalar
    TensorWrapper operator+ (TensorWrapper* tensor);

private:
    // The tensor itself is represented by a pointer to a Matrix of type Eigen::MatrixXd
    // so basically you will try to represent a tensor of arbitrary dimensions as a bunch of
    // 2D Eigen::MatrixXd matrices to use the benefit of already defined operations.
    Eigen::MatrixXd *tensor;


    //Probably more to add here.


};


#endif //LIBDL_TENSORWRAPPER_H
