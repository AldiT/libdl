
#include <iostream>
#include "catch.hpp"
#include "TensorWrapper.h"
#include "Layer.h"

#include "Eigen/Dense"


//Add tests for tensor wrapper here.


SCENARIO("Testing the experimental TensorWrapper", "[TensorWrapper_Exp]"){

    GIVEN("A TensorWrapper_Exp object") {
        libdl::TensorWrapper_Exp twe(3, 28, 28, 1, false);
        libdl::layers::Convolution2D conv("For padding", 3, 3, 1, 1);


        WHEN("we slice 2D matrix") {
            auto slice = twe.get_slice(1, 0);

            THEN("check the slices shape") {
                REQUIRE(slice.rows() == 28);
                REQUIRE(slice.cols() == 28);
            }
        }

        WHEN("we update the slice to a constant matrix, [1]") {
            Eigen::MatrixXd tmp(28, 28);
            tmp = Eigen::MatrixXd::Constant(28, 28, 1);
            twe.update_slice(1, 0, tmp);

            THEN("check that the slice updated is equal to the slice created up") {
                REQUIRE(twe.get_slice(1, 0) == tmp);
            }
        }


        WHEN("we perform correlation with random filters") {
            libdl::TensorWrapper_Exp filters(16, 3, 3, 1, true);
            twe = conv.pad(twe);
            auto correlation = twe.correlation(filters, 1);


            THEN("Check the shapes of the outputs") {//These are very dumb tests
                REQUIRE(correlation.get_batch_size() == 3);
                REQUIRE(correlation.get_tensor_height() == 28);
                REQUIRE(correlation.get_tensor_width() == 28);
                REQUIRE(correlation.get_tensor_depth() == 16);
            }
        }

        WHEN("the tensor is constant") {
            

            libdl::TensorWrapper_Exp input(3, 28, 28, 1, false);
            input.set_tensor(Eigen::MatrixXd::Constant(3, 28 * 28, 1), 28, 28, 1);
            

            libdl::TensorWrapper_Exp filters(16, 3, 3, 1, true);
            filters.set_tensor(Eigen::MatrixXd::Constant(16, 3 * 3, 1), 3, 3, 1);

            libdl::TensorWrapper_Exp output(3, 28, 28, 16, false); //same padding=1
            input = conv.pad(input);
            output = input.correlation(filters, 1);

<<<<<<< HEAD
            //std::cout << "\nFirst slice: \n" << output.get_slice(0, 0) << std::endl;
=======
>>>>>>> bb4a4c91a2e0d278c2129e2b75a7e74574f0fa81

            THEN("now check the values of the correlation") {
                auto a_slice = output.get_slice(0, 0);

                //corners
                REQUIRE(a_slice(0, 0) == 4);
                REQUIRE(a_slice(0, a_slice.cols() - 1) == 4);
                REQUIRE(a_slice(a_slice.rows() - 1, 0) == 4);
                REQUIRE(a_slice(a_slice.rows() - 1, a_slice.cols() - 1) == 4);

                //borders
                double sum = a_slice.block(0, 1, 1, a_slice.cols() - 2).sum();
                sum += a_slice.block(a_slice.rows() - 1, 1, 1, a_slice.cols() - 2).sum();
                sum += a_slice.block(1, 0, a_slice.rows() - 2, 1).sum();
                sum += a_slice.block(1, a_slice.cols() - 1, a_slice.rows() - 2, 1).sum();

                REQUIRE(sum == 6 * 26 * 4);

                //inside part
                REQUIRE(a_slice.block(1, 1, a_slice.rows() - 2, a_slice.cols() - 2).sum() ==
                        9 * (a_slice.rows() - 2) * (a_slice.cols() - 2));
            }
        }

        WHEN("Tensor is given constant") {/*
            libdl::layers::MaxPool pool(2, 2);

            Eigen::MatrixXd m(4, 4);
            m <<    2, 3, 4, 5,
                    2, 3, 4, 5,
                    2, 3, 4, 5,
                    2, 3, 4, 5;


            libdl::TensorWrapper_Exp input(4, 4, 4, 1, false);
            input.set_tensor(m, 4, 4, 1);

            libdl::TensorWrapper_Exp t(2, 2, 2, 1, false);

            t = pool.forward(input);

            std::cout << t.get_tensor() << std::endl;*/
            REQUIRE(2 == 2);
        }

    }
    /*

    GIVEN("Some layers"){
        WHEN("A maxpool layer is given an input"){
            std::cout << "Inside";
            libdl::layers::MaxPool pool1(2, 2);
            Matrixd input(4, 4);
            input << 1, 2, 3, 4,
                    1, 2, 3, 4,
                    1, 2, 3, 4,
                    1, 2, 3, 4;

            Matrixd expected_output(2, 2);
            expected_output << 2, 4,
                    2, 4;

            libdl::TensorWrapper_Exp input_tensor(1, 4, 4, 1);
            input_tensor.set_tensor(input, 4, 4, 1);
            libdl::TensorWrapper_Exp out(1, 2,2, 1);
            out.set_tensor(expected_output, 2, 2, 1);

            std::cout << "Before";
            std::cout << "Output: \n" << pool1.forward(input_tensor).get_tensor() << std::endl;
            REQUIRE((pool1.forward(input_tensor).get_tensor()) == (out.get_tensor()));
        }
    }*/



}
