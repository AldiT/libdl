//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSORWRAPPER_H
#define LIBDL_TENSORWRAPPER_H

#include "Eigen/Dense"
#include <list>
#include <vector>
#include <cstdarg>


namespace libdl{
    class TensorWrapper3D;
}


class libdl::TensorWrapper3D {

public:
    TensorWrapper3D(int first_dim, int second_dim, int third_dim);//Done
    TensorWrapper3D(const TensorWrapper3D&);//Check TODO
    TensorWrapper3D& operator=(const TensorWrapper3D&);//CHeck TODO

    ~TensorWrapper3D();//Not Done: But there is nothing to be done actually


    ///
    //// Indexing operator overload, same notation as Eigen
    ////
    //
    Eigen::MatrixXd operator() (int i);//Done
    Eigen::VectorXd operator() (int i, int j);//Done
    double          operator() (int i, int j, int z);//Done


    TensorWrapper3D operator* (TensorWrapper3D&);   //Multiply two tensors with compatible sizes //Done
    TensorWrapper3D operator* (double);             //Multiply each element of the tensor with a scalar //Done
    TensorWrapper3D operator+ (TensorWrapper3D&); //Done

    int get_first_dim () const ;//Done
    int get_second_dim() const ;//Done
    int get_third_dim () const ;//Done

    int size();


    static double           DotProduct(TensorWrapper3D&, TensorWrapper3D&);//Done
    static TensorWrapper3D  ElementWiseMult(TensorWrapper3D&, TensorWrapper3D&);//Done

private:
    // The tensor itself is represented by a pointer to a Matrix of type Eigen::MatrixXd
    // so basically you will try to represent a tensor of arbitrary dimensions as a bunch of
    // 2D Eigen::MatrixXd matrices to use the benefit of already defined operations.
    std::vector<Eigen::MatrixXd> tensor;
    int                          first_dim;
    int                          second_dim;
    int                          third_dim;

    //Probably more to add here.


};


#endif //LIBDL_TENSORWRAPPER_H
