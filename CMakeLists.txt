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



function(SET_PATHS_TO_LIBRARIES)
    set(Catch2_H \"${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/include/catch.hpp\" PARENT_SCOPE)
    message("Paths to Catch2 set! Use Catch2_H for catch.hpp")

    include_directories("${PROJECT_SOURCE_DIR}/extern/eigen3")
    message("Eigen3 path set up!")

    include_directories("${PROJECT_SOURCE_DIR}/extern/pybind11/include")
endfunction()

#set_paths_to_libraries()
#add_definitions(-DCatch2_H=${Catch2_H})



add_executable(${TEST_TARGET} main.cpp)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
