//
// Created by Aldi Topalli on 2019-06-16.
//

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

int add(uint* arr){
    int length = sizeof(arr) / sizeof(uint);

    std::cout << "\n\n";
    for(int i = 0; i < length; i++){
        std::cout << arr[i] << " ";
    }
    std::cout << "\n\n";

    return 1;
}

PYBIND11_MODULE(example, m){

    m.def("add", &add, "");
}