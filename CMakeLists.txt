cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)


set(TEST_TARGET libdl)

set(test_VERSION_MINOR 0)
set(test_VERSION_MAJOR 1)

set(CMAKE_CXX_STANDARD 17)




#set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2)
#add_library(Catch2::Catch IMPORTED INTERFACE)
#set_property(TARGET Catch2::Catch PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CATCH_INCLUDE_DIR}")

include_directories("include")
#include_directories("extern")
#include_directories("/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/include/python3.7m")




function(SET_PATHS_TO_LIBRARIES)
    set(Catch2_H \"${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/include/catch.hpp\" PARENT_SCOPE)
    message("Paths to Catch2 set! Use Catch2_H for catch.hpp")

    include_directories("${PROJECT_SOURCE_DIR}/extern/eigen3")
    message("Eigen3 path set up!")

    include_directories("${PROJECT_SOURCE_DIR}/extern/pybind11/include")
endfunction()

#set_paths_to_libraries()
#add_definitions(-DCatch2_H=${Catch2_H})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Catch2 REQUIRED)
include_directories(${CATCH2_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(PYBIND11_CPP_STANDARD -std=c++14)
set(PYBIND11_CPP_STANDARD -std=c++1z) # Experimental C++17 support
# MSVC:
set(PYBIND11_CPP_STANDARD /std:c++14)
set(PYBIND11_CPP_STANDARD /std:c++latest) # Enables some MSVC C++17 features
if (PYBIND11_TESTS)
    add_subdirectory(pybind11)
endif ()

include_directories(${PYBIND11_INCLUDE_DIRS})

add_executable(${TEST_TARGET} src/Layer/Layer.cpp include/Layer.h src/Model/Model.cpp include/Model.h)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

target_link_libraries(libdl ${PYTHON_LIBRARIES})
target_link_libraries(libdl Eigen3::Eigen)
target_link_libraries(libdl Catch2::Catch2)
target_link_libraries(libdl pybind11::module)

