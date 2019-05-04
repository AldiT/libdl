cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)


set(TEST_TARGET libdl)

set(test_VERSION_MINOR 0)
set(test_VERSION_MAJOR 1)

set(CMAKE_CXX_STANDARD 17)

add_executable(${TEST_TARGET} main.cpp TutorialConfig.cpp TutorialConfig.h)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

