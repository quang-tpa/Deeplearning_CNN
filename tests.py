# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:59:03 2020

@author: Admin
"""

import tensorflow as tf
from networks import FC_2Layer,Conv3_Layer
from mnist import mnist
import numpy as np
import ops

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test_fc():

    # Load data
    x_train, t_train, x_test, t_test = mnist.load()

    # Normalize data
    X_train = np.array(x_train, dtype=np.float64)
    X_test = np.array(x_test, dtype=np.float64)

    X_train -= int(np.mean(X_train))
    X_train /= int(np.std(X_train))

    X_test -= int(np.mean(X_test))
    X_test /= int(np.std(X_test))


    # Initialize network
    network = FC_2Layer()


    # Train Network
    epochs = 20
    for e in range(epochs):
        total_loss = 0
        correct = 0
        for i, img in enumerate(X_train):

            ### Forward Pass ###
            y_hat = network.forward(img)

            ### Cross Entropy Loss ###
            y = t_train[i]
            loss, y = ops.cross_entropy(y, y_hat)
            total_loss += loss

            # Calculate Accuracy
            if list(y_hat).index(max(y_hat)) == list(y).index(max(y)):
                correct +=1

            ### Backward Pass ###
            network.backward(y)

            if i % 5000 == 0:
                print("Epoch " + str(e) + ", Sample " + str(i) + ", Loss: " + str(loss)[:9] + ", ACC: " + str(correct/(i+1)))
                #print("time elapsed: " + str(time.time() - start_time))


        print("Epoch " + str(e) + " Loss: " + str(total_loss/len(X_train))[:9] + " Acc: " + str(correct/len(X_train)))

import utils
import time
import pickle
def train_lenet():

    X_train, X_test, t_train, t_test = utils.load_mnist('square')
    
    # Initilize network
    #network = LeNet5()
    #network = Conv2_Layer()
    network = Conv3_Layer()
    #return
    #network = Conv4_Layer()
   
    # Train Network
    epochs = 20
    batch_size = 100

    start_time = time.time()
    for e in range(epochs):

        total_loss = 0
        correct = 0
        correct_this_batch = 0

        for i, img in enumerate(X_train):

            y = t_train[i]
  

            ### Forward Pass ###
            y_hat = network.forward(img)

            ### Cross Entropy Loss ###
            
            loss, y = ops.cross_entropy(y, y_hat)
            total_loss += loss

            # Calculate Accuracy
            if list(y_hat).index(max(y_hat)) == list(y).index(max(y)):
                correct +=1
                correct_this_batch += 1
            

            ### Backward Pass ###
            network.backward(y)

            
            if i % batch_size == 0:
                print("Epoch " + str(e) + ", Sample " + str(i) + ", Loss: " + str(loss)[:6] + ", ACC: " + str(correct/(i+1))[0:6] + ", Batch ACC: " + str(correct_this_batch/batch_size))
                print("Learning Rate: " + str(network.learning_rate))
                print("time elapsed: " + str(time.time() - start_time))
                correct_this_batch = 0
                
                #if i % 1000 == 0:    
                #    network.learning_rate = network.learning_rate * 0.98
                    

                obj = []
                for layer in network.layers:
                    obj.append(layer.extract())


        print("Epoch " + str(e) + " Loss: " + str(total_loss/len(X_train))[:9] + " Acc: " + str(correct/len(X_train)))

train_lenet()