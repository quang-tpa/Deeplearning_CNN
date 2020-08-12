# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:53:25 2020

@author: Admin
"""
import numpy as np


#================   Conv2D  =================
class Conv2D:
    def __init__(self, num_channels, num_filters, kernel_size, stride, learning_rate, name):

        self.name = name
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_channels = num_channels
        # Weight initialization according to Andrew Ng's Deeplearning.ai course: https://www.youtube.com/watch?v=s2coXdufOzE
        self.weights = np.random.random((self.num_filters, self.num_channels, self.kernel_size,self.kernel_size)) * np.sqrt((2. / (self.num_channels * self.kernel_size * self.kernel_size)))
        self.bias = np.zeros((self.num_filters, 1))

        self.inputs = None

        print(self.name + " min weight: " + str(min(self.weights.ravel())))
        print(self.name + " max weight: " + str(max(self.weights.ravel())))


    def forward(self, inputs):

        #print(self.name)
        self.inputs = inputs
        input_size = len(inputs)
        # Feauture map to be returned
        output_size = int(((input_size - self.kernel_size)/self.stride)+1)
        output = np.zeros((output_size, output_size, self.num_filters))
        
        
        for f in range(len(self.weights)):
            for r in range(len(self.inputs)):
                for c in range(len(self.inputs[0])):
                    
                    # subimg
                    subimg = self.inputs[r:r+self.kernel_size, c:c+self.kernel_size, :]
      
                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue
                    
                    # output = subimg*W + b
                    output[r][c][f] = np.sum(subimg.T * self.weights[f,:,:,:]) + self.bias[f]

        return output

    def backward(self, dy):
        #print("====== " + str(self.name) + " Backward ======")
        dw = np.zeros((self.weights.shape))
        dx = np.zeros(self.inputs.shape)
        db = np.zeros(self.bias.shape)

        rows, cols, filters = dy.shape

        # Perform Convolution to determine dw and dx
        for f in range(filters):
            for r in range(rows):
                for c in range(cols):
                    subimg = self.inputs[r:r+self.kernel_size, c:c+self.kernel_size, :]

                    # Valid Padding
                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue

                    dw[f,:,:,:]  += dy[r,c,f] * subimg.T
                    dx[r:r+self.kernel_size, c:c+self.kernel_size, :] += dy[r,c, f] * self.weights[f,:,:,:].T
        
        # Calculate db
        for f in range(filters):
            db[f] = np.sum(dy[:, :, f])

        # Update weights and biases

        old_weights = np.copy(self.weights)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return dx


    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
        
#================   MaxPool  =================
class MaxPool:
    def __init__(self, kernel_size, stride, name):
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        self.inputs = None


    def forward(self, inputs):

        self.inputs = inputs
        num_filters = inputs.shape[2]
        input_size = len(inputs)
        # Feauture map to be returned
        output_size = int(  ((input_size - self.kernel_size) / self.stride ) + 1   )
        output = np.zeros((output_size, output_size, num_filters))

        # Iterate trough every dimension (channel) of the input
        for d in range(num_filters):
            # Iterate through every row, enumerate ánh xạ (max subimg) của input sang output
            for i, row in enumerate(range(0, len(inputs), self.stride)):
                # Iterate through every column
                for j, col in enumerate(range(0, len(inputs[0]), self.stride)):

                    # Sub section of the input to be convolved (cross-correlated)
                    # Essentialy a sliding window accross the input
                    subimg = inputs[row : row + self.kernel_size, col : col + self.kernel_size, d]

                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue

                    # Max Pooling
                    result = np.max(subimg)
                    output[i][j][d] = result
        

        return output

    def backward(self, dy):

        num_filters = self.inputs.shape[2]
        input_size = len(self.inputs)

        # Output will be the same shape as the input to the forward pass
        dx = np.zeros(self.inputs.shape)

        # Iterate trough every dimension (channel) of the input
        for d in range(num_filters):
            # Iterate through every row
            for i, row in enumerate(range(0, len(self.inputs), self.stride)):
                # Iterate through every column
                for j, col in enumerate(range(0, len(self.inputs[0]), self.stride)):

                    dx[i][j][d] = dy[i // self.kernel_size][ j // self.kernel_size ][d]
        
        # Returning feature map
        return dx

    def extract(self):
        return

class FullyConnected:

    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.name = name

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.random((self.num_inputs,self.num_outputs)) * np.sqrt((2. / (self.num_inputs * self.num_outputs)))
        print(self.name + " min weight: " + str(min(self.weights.ravel())))
        print(self.name + " max weight: " + str(max(self.weights.ravel())))
        
        self.bias = np.zeros((self.num_outputs,1))

        self.inputs = None
        self.activation = None

        self.learning_rate = learning_rate


    def forward(self, inputs):
        self.inputs = inputs
        self.activation = np.dot(inputs, self.weights) + self.bias.T
        return self.activation.ravel()

    def backward(self, dy):
        
        self.inputs = self.inputs.ravel()
        self.inputs = np.expand_dims(self.inputs, axis = 1)


        dy = dy.ravel()
        dy = np.expand_dims(dy, axis = 1)
       
        
        # Get the dot product of the error (dy) given the input
        # This will give you the gradients corresponding to each weight
        dw = np.dot(self.inputs, dy.T)
        db = np.sum(dy, axis=1, keepdims=True)
       

        #dx = np.dot(dy.T, self.weights.T)
        dx = np.dot(self.weights, dy)

        
        old_weights = np.copy(self.weights)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Flatten:
    def __init__(self):
        self.name = "Flatten"
        self.inputs = None
    def forward(self, inputs):
        self.inputs = inputs
        self.C, self.W, self.H = inputs.shape
        output = inputs.reshape(1, self.C*self.W*self.H)

        return output

    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)

    def extract(self):
        return
    
    
class ReLu:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    def extract(self):
        return


class Sigmoid:
    def __init__(self):
        self.name = "Sigmoid"
        self.inputs = None

    def sigmoid_backend(x):
        """Applies the sigmoid function elementwise."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward(self, inputs):
        self.inputs = inputs
        x = inputs
        #self.activation = (1.0 / (1 + np.exp(-self.inputs) ))
        self.activation = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        print("Sigmoid Forward max: " + str(np.max(self.activation)))
        print("Sigmoid Forward min: " + str(np.min(self.activation)))
        return self.activation

    def backward(self, dy):

        def sigmoid_backend(x):
            """Applies the sigmoid function elementwise."""
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        dx = sigmoid_backend(dy) * (1 - sigmoid_backend(dy))
        
        if dx.shape[1] == 1:
            dx = np.expand_dims(dx, axis = 1)

        return dx

    def extract(self):
        return

class Softmax():
    def __init__(self):
        self.name = "Softmax"
        self.activation = None


    def forward(self, inputs):

        inputs = inputs.ravel()
        exp = np.exp(inputs, dtype=np.float)
        self.activation = exp/np.sum(exp)
    
        return self.activation

    def forward_stable(self, inputs):
        inputs = inputs - np.max(inputs)
        exp = np.exp(inputs - np.max(inputs), dtype=np.float128)
        self.activation = exp / np.sum(exp)
        return self.activation

    def backward(self, y_probs):

        return self.activation - y_probs

    def extract(self):
        return

# Other usefull functions
def cross_entropy(y, y_hat):

    # One hot encode Y to create a distribution
    y_probs = np.zeros(len(y_hat))
    y_probs[y] = 1.0

    return -np.sum(y_probs * np.log(y_hat)), y_probs