//
// Created by Aldi Topalli on 2019-05-10.
//

#ifndef LIBDL_TENSORWRAPPER_H
#define LIBDL_TENSORWRAPPER_H

#include "Eigen/Dense"
#include <memory>
#include <list>
#include <vector>
#include <cstdarg>


namespace libdl{
    class TensorWrapper3D;
    class TensorWrapper_Exp;
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

    Eigen::MatrixXd at(int i);//Done
    Eigen::VectorXd at(int i, int j);//Done
    double          at(int i, int j, int z);//Done


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


class libdl::TensorWrapper_Exp{
public:

    //Constructors
    TensorWrapper_Exp(int batch_size_=16, int tensor_height_=28, int tensor_width_=28, int tensor_depth_=3, bool are_filters_=false); //default values for mnist
    TensorWrapper_Exp(const TensorWrapper_Exp&);
    TensorWrapper_Exp& operator= (const TensorWrapper_Exp&);

    Eigen::MatrixXd operator() (int i) const; //return i-th instance
    const TensorWrapper_Exp operator+(TensorWrapper_Exp&) const;
    const TensorWrapper_Exp operator*(double) const;

    TensorWrapper_Exp& correlation(TensorWrapper_Exp& , int , int, TensorWrapper_Exp&) const;
    TensorWrapper_Exp& full_convolution(TensorWrapper_Exp&, TensorWrapper_Exp&) const;




    void from_EigenMatrixXd(const Eigen::MatrixXd& matrix_);
    Eigen::MatrixXd to_EigenMatrixXd();


    Eigen::MatrixXd get_slice(int instance_, int depth_) const;
    void            update_slice(int instance_, int depth_, Eigen::MatrixXd new_slice_);

    //getters_setters
    int get_batch_size() const;
    int get_tensor_height() const;
    int get_tensor_width() const;
    int get_tensor_depth() const;
    bool is_filter() const;

    Eigen::MatrixXd& get_tensor() const;
    void set_tensor(Eigen::MatrixXd new_tensor);

    friend std::ostream& operator<< (std::ostream& os, TensorWrapper_Exp wrapper_){
        os << "Tensor: \n" << wrapper_.get_tensor() << std::endl;
    }

protected:

    static Eigen::MatrixXd correlation2D(Eigen::MatrixXd& m1, Eigen::MatrixXd& m2, int, int stride=1);
    static Eigen::MatrixXd pad(Eigen::MatrixXd&, int);

private:
    std::unique_ptr<Eigen::MatrixXd> tensor;
    int batch_size;
    int tensor_height;
    int tensor_width;
    int tensor_depth;
    bool are_filters;
};



#endif //LIBDL_TENSORWRAPPER_H
