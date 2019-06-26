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
 In **line 65** is specified the batch size, in **line 110** and **111** are specified how many batches to train on(b<10)
 this due to the fact that the library is painfully slow for now... and also here is where the number of epochs is also specified.

 Suggestion: As I mentioned, the library is very slow for the moment, so to prove that it is actually training, I would
 use overfiting on a single image. To do that you need to set epoch on **line 110** to smth like ~10 and set the batch_limit
 provisional condition from 10 to 1 on (**line 65**), also batch_size on **line 65** to 1.


 ## Find main function on: tests/Layer_tests.cpp