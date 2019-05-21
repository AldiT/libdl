# libdl

## How to test the XOR problem

### I will try to be as clear as possible in case something is misunderstood or is not going right please let me know

* Download the master branch as a zip (or whatever).
* Unzip the folder and run the following commands
* ```mkdir build```
* ```cd build```
* ```cmake ..```
* ```make```
* And execute ```./libdl```



### The whole program is set up on the Layer_tests.cpp file under tests folder (this is where **main** function is)
 The points which are being predicted are also specified in the main function, if you feel that you need to test new points
 at the bottom of the main function there is the test phase in **line 96-122**. The input is also specified in these lines
 so to test new points just change them as you like.

 ## Find main function on: tests/Layer_tests.cpp