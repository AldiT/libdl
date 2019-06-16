cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)


set(TEST_TARGET libdl)


set(CMAKE_CXX_STANDARD 17)




#set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2)
#add_library(Catch2::Catch IMPORTED INTERFACE)
#set_property(TARGET Catch2::Catch PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CATCH_INCLUDE_DIR}")

include_directories("include")
include_directories("extern")
include_directories("extern/pybind11/include")




function(SET_PATHS_TO_LIBRARIES)
    set(Catch2_H \"${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/include/catch.hpp\" PARENT_SCOPE)
    message("Paths to Catch2 set! Use Catch2_H for catch.hpp")

    include_directories("${PROJECT_SOURCE_DIR}/extern/eigen3")
    message("Eigen3 path set up!")

    include_directories("${PROJECT_SOURCE_DIR}/extern/pybind11/include")

endfunction()



find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/opencv)
include_directories("${PROJECT_SOURCE_DIR}/extern/opencv/include")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(pybind11 INTERFACE -fsized-deallocation)
endif()

pybind11_add_module(libdl "${PROJECT_SOURCE_DIR}/tests/Layer_tests.cpp")

add_executable(${TEST_TARGET} src/Layer/Layer.cpp include/Layer.h src/Model/Model.cpp include/Model.h tests/Layer_tests.cpp src/TensorWrapper.cpp include/TensorWrapper.h src/ErrorFunctions.cpp include/ErrorFunctions.h)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

target_link_libraries(libdl ${PYTHON_LIBRARIES})
target_link_libraries(libdl Eigen3::Eigen)

