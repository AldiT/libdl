# libdl

## How to test the MNIST problem

### I will try to be as clear as possible in case something is misunderstood or is not going right please let me know

* Download the master branch as a zip (or whatever).
* Unzip the folder and run the following commands
* ```git clone https://github.com/pybind/pybind11.git extern/pybind11 --recursive```
* ```mkdir build```
* ```cd build```
* ```cmake -DCMAKE_BUILD_TYPE=Release ..```
* ```make```
* To run the tests: ```./run_tests```
* A dynamic library .so in my Unix system should be created , copy it to the root directory of the project (/libdl)



### The tests are runed on the Layer_tests.cpp file under tests folder (this is where **main** function is)
On folder python_scripts there is a jupyter notebook , open it with jupyter and see the small demo I have set up there.

I will be reachable by email all day if you have any further questions.


 ## Find main function on: tests/Layer_tests.cpp