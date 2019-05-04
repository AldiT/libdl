#include <iostream>
#include <functional>

#include "TutorialConfig.h"

int main()
{
    [out = std::ref(std::cout << "Hello")](){out.get() << " World!\n";}();

    TutorialConfig t;
    t.test_func();
}

