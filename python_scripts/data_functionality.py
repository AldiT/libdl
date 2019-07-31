import torch
import numpy as np
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper
import matplotlib.pyplot as plt
import os



#This data loader will be built specifically for the Malaria dataset project

class DataLoader():
    def __init__(self, path, batch_size):
        pass

    #Get one batch of data
    #Transform it and send it to the network
    def get_batch(self):
        pass
    
    