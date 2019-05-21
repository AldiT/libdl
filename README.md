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
 just add some rows to the end of the input variable of type Eigen::MatrixXd.

 ## Or

 You can create a new Eigen::MatrixXd variable at the end of the main function (where it is tested, right before the
  cout << "Output") and feed it to the test run **lines 98-103** file : tests/Layers_tests.cpp