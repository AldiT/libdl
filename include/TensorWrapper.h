//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSORWRAPPER_H
#define LIBDL_TENSORWRAPPER_H

#include "Eigen/Dense"



namespace libdl{
    class TensorWrapper;
}


class libdl::TensorWrapper {

public:
    TensorWrapper();
    TensorWrapper(Eigen::MatrixXd tensor);
    TensorWrapper(int );

    TensorWrapper(int* dims);
    TensorWrapper(TensorWrapper* tensor);

    ~TensorWrapper();


    ///
    //// Indexing operator overload, same notation as Eigen
    ////
    //
    TensorWrapper* operator() (int i);
    TensorWrapper* operator() (int i, int j);
    TensorWrapper* operator() (int i, int j, int z);
    TensorWrapper* operator() (int b, int i, int j, int z);


    TensorWrapper operator* (TensorWrapper* tensor);//Multiply two tensors with compatible sizes
    TensorWrapper operator+ ();//Multiply each element of the tensor with a scalar
    TensorWrapper operator+ (TensorWrapper* tensor);

private:
    // The tensor itself is represented by a pointer to a Matrix of type Eigen::MatrixXd
    // so basically you will try to represent a tensor of arbitrary dimensions as a bunch of
    // 2D Eigen::MatrixXd matrices to use the benefit of already defined operations.
    Eigen::MatrixXd *tensor;
    int third_dimension;


    //Probably more to add here.


};


#endif //LIBDL_TENSORWRAPPER_H
