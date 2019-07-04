# libdl

## How to test the MNIST problem

### I will try to be as clear as possible in case something is misunderstood or is not going right please let me know

* Download the master branch as a zip (or whatever).
* Unzip the folder and run the following commands
* ```mkdir build```
* ```cd build```
* ```cmake ..```
* ```make```
* And execute ```./libdl```



### The whole program is set up on the Layer_tests.cpp file under tests folder (this is where **main** function is)
 In **line 79** are specified some of the hyperparameters(batch_size, batch_limit, lr).
 I train only on a small batch, this due to the fact that the library is painfully slow for now... so batch_limit is a provisional hyperparameter due to above mentioned circumctances.

 On my computer I reach up to 0.9 accuracy on the batch after ~10 minutes.
 There should be three phases of output:
    1. Before training
    2. Training
    3. Test

I will be reachable by email all day if you have any further questions.


 ## Find main function on: tests/Layer_tests.cpp