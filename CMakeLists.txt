cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)


set(TEST_TARGET run_tests)


set(CMAKE_CXX_STANDARD 17)

#Remove this two lines before doing anything else
set(PYTHON_LIBRARY "/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib")
set(PYTHON_INCLUDE_DIR "/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/include/python3.7m")



#set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2)
#add_library(Catch2::Catch IMPORTED INTERFACE)
#set_property(TARGET Catch2::Catch PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CATCH_INCLUDE_DIR}")

include_directories("include")
include_directories("extern")
include_directories("extern/pybind11/include")
include_directories("extern/eigen3")




function(SET_PATHS_TO_LIBRARIES)
    set(Catch2_H \"${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/include/catch.hpp\" PARENT_SCOPE)
    message("Paths to Catch2 set! Use Catch2_H for catch.hpp")

    include_directories("${PROJECT_SOURCE_DIR}/extern/eigen3")
    message("Eigen3 path set up!")

endfunction()


find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

message(${PYTHON_INCLUDE_DIRS})

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#include_directories(${EIGEN3_INCLUDE_DIRS})


add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(pybind11 INTERFACE -fsized-deallocation)
endif()



add_executable(${TEST_TARGET} src/Layer/Layer.cpp include/Layer.h src/Model/Model.cpp include/Model.h tests/Layer_tests.cpp src/TensorWrapper.cpp include/TensorWrapper.h src/ErrorFunctions.cpp include/ErrorFunctions.h include/data.h include/data_handler.h src/data.cpp src/data_handler.cpp tests/tests.cpp)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
target_link_libraries(${TEST_TARGET} ${PYTHON_LIBRARIES})

add_library(lib src/Layer/Layer.cpp include/Layer.h src/Model/Model.cpp include/Model.h tests/Layer_tests.cpp src/TensorWrapper.cpp include/TensorWrapper.h src/ErrorFunctions.cpp include/ErrorFunctions.h include/data.h include/data_handler.h src/data.cpp src/data_handler.cpp tests/tests.cpp)

pybind11_add_module(libdl "${PROJECT_SOURCE_DIR}/tests/pybind_tests.cpp")
target_link_libraries(libdl PRIVATE lib)

#target_link_libraries(${TEST_TARGET} Eigen3::Eigen)

