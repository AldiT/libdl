import torch
import numpy as np
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper
import matplotlib.pyplot as plt
import os
"""
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize(255), 
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder()

#This data loader will be built specifically for the Malaria dataset project

class DataLoader():
    def __init__(self, path, batch_size):
        pass

    #Get one batch of data
    #Transform it and send it to the network
    def get_batch(self):
        pass
"""

import numpy as np
#Download mnist and feed it to the network
from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print(train_data.shape)

train_data = np.reshape(train_data, (train_data.shape[0], -1))





class Convolution:
    def __init__(self, num_filters, kernel_size, input_depth, padding):
        self.filters = np.random.rand(num_filters, kernel_size, kernel_size, input_depth)
        self.padding = padding
        self.input = None

    def forward(self, incoming):
        self.incoming = incoming
    
        self.padded_input = np.zeros((incoming.shape[0], incoming.shape[1] + 2 * self.padding,
            incoming.shape[2] + 2 * self.padding, incoming.shape[3]))


        self.padded_input[:, self.padding:, self.padding:, :] = incoming

        o_rows = incoming.shape[1] - self.filters.shape[1] + 1
        o_cols = incoming.shape[2] - self.filters.shape[2] + 1

        output = np.zeros((incoming.shape[0], o_rows, o_cols, self.filters.shape[0]), dtype="float64")

        for instance in range(incoming.shape[0]):
            for kernel in range(self.filters.shape[0]):
                for x in range(output.shape[1]):
                    for y in range(output.shape[2]):
                        output[instance, x, y, kernel] = np.multiply(self.padded_input[instance, x:x+self.filters.shape[1], y:y+self.filters.shape[2], :], self.filters[kernel]).sum()

        return output

    def convolution(self, incoming, filters):

        o_rows = incoming.shape[1] - filters.shape[1] + 1
        o_cols = incoming.shape[2] - filters.shape[2] + 1

        output = np.zeros((incoming.shape[0], o_rows, o_cols, filters.shape[0]), dtype="float64")

        for instance in range(incoming.shape[0]):
            for kernel in range(filters.shape[0]):
                for x in range(output.shape[0]):
                    for y in range(output.shape[1]):
                        output[instance, x, y, kernel] = np.multiply(incoming[instance, x:x+kernel.shape[0], y:y+kernel.shape[1], :], filters[kernel]).sum()

        return output

    def backward(self, gradients, lr):
        
        filter_grads = self.filter_conv(gradients)
        self.filters -= lr * filter_grads

        input_grads = self.input_conv(gradients)

        return input_grads


    def input_conv(self, gradients):
        
    
        reshaped_filters = np.zeros((self.filters.shape[3], self.filters.shape[1], 
            self.filters.shape[2], self.filters.shape[0]))

        for f in range(self.filters.shape[0]):
            for s in range(reshaped_filters.shape[3]):
                reshaped_filters[s, :, :, f] = self.filters[f, :, :, s]

        o_rows = gradients.shape[1] - reshaped_filters.shape[1] + 1
        o_cols = gradients.shape[2] - reshaped_filters.shape[2] + 1

        output = np.zeros((gradients.shape[0], o_rows, o_cols, reshaped_filters.shape[0]), dtype="float64")

        for instance in range(gradients.shape[0]):
            for kernel in range(reshaped_filters.shape[0]):
                for x in range(output.shape[0]):
                    for y in range(output.shape[1]):
                        output[instance, x, y, kernel] = np.multiply(gradients[instance, x:x+kernel.shape[0], y:y+kernel.shape[1], :], reshaped_filters[kernel]).sum()


        return output
        
    

    def filter_conv(self, gradients):

        output = np.zeros_like(self.filters)

        for kernel in range(self.filters.shape[0]):
            for gradient_slice in range(gradients.shape[0]):
                for x in range(self.filters.shape[1]):
                    for y in range(self.filters.shape[2]):
                        for input_slice in range(self.padded_input.shape[3]):
                            for instance in range(gradients.shape[0]):
                                output[kernel, x, y, input_slice] += np.multiply(gradients[instance, :, :, gradient_slice], self.padded_input[instance, x:x+gradients.shape[1], y:y+gradients.shape[2], input_slice]).sum()


        return output
        
    


if __name__ == "__main__":
    ones = np.ones((1, 4, 4, 1), dtype='float64')
    kernel = np.ones((1, 3, 3, 1), dtype='float64')

    
        
    


    conv = Convolution(1, 3, 1, 0)
    conv.filters = kernel

    out = conv.forward(incoming = ones)
