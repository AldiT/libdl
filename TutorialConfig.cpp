//
// Created by Aldi Topalli on 2019-05-03.
//

#include "TutorialConfig.h"
#include <iostream>


void TutorialConfig::test_func() {
#ifndef test_VERSION_MAJOR
    std::cout << "From the test function " << std::endl;
#endif
}