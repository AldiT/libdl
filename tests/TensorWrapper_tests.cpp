
#include <iostream>
#include "catch.hpp"
#include "TensorWrapper.h"

#include "Eigen/Dense"


//Add tests for tensor wrapper here.


SCENARIO("Testing the experimental TensorWrapper", "[TensorWrapper_Exp]"){

    GIVEN("A TensorWrapper_Exp object"){
        libdl::TensorWrapper_Exp twe(3, 28, 28, 1, false);


        WHEN("we slice 2D matrix"){
            auto slice = twe.get_slice(1, 0);

            THEN("check the slices shape") {
                REQUIRE(slice.rows() == 28);
                REQUIRE(slice.cols() == 28);
            }
        }

        WHEN("we update the slice to a constant matrix, [1]"){
            Eigen::MatrixXd tmp (28, 28);
            tmp = Eigen::MatrixXd::Constant(28, 28, 1);
            twe.update_slice(1, 0, tmp);

            THEN("check that the slice updated is equal to the slice created up") {
                REQUIRE(twe.get_slice(1, 0) == tmp);
            }
        }


        WHEN("we perform correlation with random filters"){
            libdl::TensorWrapper_Exp filters(16, 3, 3, 1, true);
            libdl::TensorWrapper_Exp correlation(3, 28, 28, 16, false);
            twe.correlation(filters, 1, 1, correlation);

            THEN("Check the shapes of the outputs") {//These are very dumb tests
                REQUIRE(correlation.get_batch_size()    ==  3);
                REQUIRE(correlation.get_tensor_height() == 28);
                REQUIRE(correlation.get_tensor_width()  == 28);
                REQUIRE(correlation.get_tensor_depth()  == 16);
            }
        }

        WHEN("the tensor is constant"){
            libdl::TensorWrapper_Exp input(3, 28, 28, 1, false);
            input.set_tensor(Eigen::MatrixXd::Constant(3, 28*28, 1), 28, 28, 1);

            libdl::TensorWrapper_Exp filters(16, 3, 3, 1, true);
            filters.set_tensor(Eigen::MatrixXd::Constant(16, 3*3, 1), 3, 3, 1);

            libdl::TensorWrapper_Exp output(3, 28, 28, 16, false); //same padding=1
            input.correlation(filters, 1, 1, output);


            THEN("now check the values of the correlation"){
                auto a_slice = output.get_slice(0, 0);

                //corners
                REQUIRE(a_slice(0, 0) == 4);
                REQUIRE(a_slice(0, a_slice.cols()-1) == 4);
                REQUIRE(a_slice(a_slice.rows()-1, 0) == 4);
                REQUIRE(a_slice(a_slice.rows()-1, a_slice.cols()-1) == 4);

                //borders
                double sum = a_slice.block(0, 1, 1, a_slice.cols()-2).sum();
                sum += a_slice.block(a_slice.rows()-1, 1, 1, a_slice.cols()-2).sum();
                sum += a_slice.block(1, 0, a_slice.rows()-2, 1).sum();
                sum += a_slice.block(1, a_slice.cols()-1, a_slice.rows()-2, 1).sum();

                REQUIRE(sum == 6*26*4);

                //inside part
                REQUIRE(a_slice.block(1, 1, a_slice.rows()-2, a_slice.cols()-2).sum() == 9*(a_slice.rows()-2) * (a_slice.cols()-2));
            }
        }
    }



}