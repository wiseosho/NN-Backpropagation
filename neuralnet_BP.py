#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:10:48 2017

@author: june
"""
"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST imbage data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library

import copy
import inputdata as inda

# Third-party libraries
import numpy as np


#%%
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def P_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def Cost_der(h, y):
    return (h-y)

#%%
class Network(object):
    
    def __init__(self, net_dim, lr=0.1, Ws=None): # net_dim = [#nodes layer, start from input nodes]
        #Create weights ans biases as a one matrix.
        if (Ws == None):
            self.weights = [np.random.randn(o, i+1) for i, o in zip(net_dim[:-1], net_dim[1:] )]
            #self.biases = [np.random.randn(o, 1) for o in net_dim[1:] ]
        else:
            self.weights = Ws
            #self.biases = Bs
        self.lr = lr
        #self.activations = np.zeros((Nlayer, ))
        #self.output = 
        
    def stochastic_gradient_descent(self,train, epochs, test=None, lr=0.5, Bsize = 200, Ws=None):
        if (Ws !=None) :
            self.weights = Ws
            #self.biases = Bs
        # Mini_batch with k partitioning will be used
        N = len(train)
        for epoch in range(epochs):

            #devide train_data_set
            #np.random.shuffle(train)
            #np.arange(0, N, partition)
            #batches = [ train[i : i+k] for i in range(0, N, Bsize)]

            np.random.shuffle(train)
            #training_shuffled = [ [training[0][i],training[1][i]] for i in idx]
            batches = [ train[i: i+Bsize] for i in range(0, N, Bsize)]

            #Learning from the batch data
            for batch in batches:
                self.update_from_batch(batch)

            # Evaluate from the test data
            if test :
                print ("Epoch {0} : {1} accuracy".format(epoch, self.evaluate(test)) )
            else:
                print ("Epoch {0} Complete. No test data available")
        return (self.weights)#, self.biases)

    def update_from_batch(self,batch):
        #batch is composed of x, y of input node, labels
        N = len(batch)
        #make placeholders for delta_parameters
        accum_delta_w = [np.zeros_like(w) for w in self.weights]
        #accum_delta_b = [np.zeros_like(b) for b in self.biases]
        
        #calculate backpropagated derivatives of Cost function w.r.t weights and biases
        # This 
        
        #This will calculate samples of whole batch at once as a matrix
        accum_delta_w = self.backpropagation_batch(batch)
        '''
        for sample in batch: # Suppose the sample consists of [input, label], where shape(input) = n,1/
            del_w = self.backpropagation(sample) # accum_delta_w, accum_delta_b #
        
            #add calculated sample into (accumulate)
            accum_delta_w = [acc_del_w + del_w for acc_del_w, del_w in zip(accum_delta_w,del_w )]
            #accum_delta_b = [acc_del_b + del_b for acc_del_b, del_b in zip(accum_delta_b, del_b)]
        '''
        
        # weight correction using accumulated delta with learning rate(lr)
        self.weights = [w - (self.lr/N)*del_w for w, del_w in zip(self.weights, accum_delta_w)]
        #self.biases = [b - (self.lr/N)*del_b for b, del_b in zip(self.biases, accum_delta_b)]
    def backpropagation(self, sample):
        #Suppose sample is One case with N feasures in shape of matrix(N,1)
        #activations and sum values of each layer should be counted and listed in proccess of feedforward 
        #
        #Layer_val = np.zeros(1,1)
        #Storing intermediate output values will be appended from existing list
        in_a = [sample[0]] #first value will be the input features
        out_z = []
        ina = copy.deepcopy(in_a[0])
        #Feed_forward and save each layer output
        for w in self.weights:#, self.biases):
            z = np.dot(w, np.vstack((ina,[1]) )) #+b
            ina = sigmoid(z)
            out_z.append(z)
            in_a.append(ina)
        #Backpropagation using parameters
        dw = [np.zeros_like(w) for w in self.weights]
        #db = [np.zeros_like(b) for b in self.biases]
        N = len(out_z)
        #1. Get Cost derivative values(dz, dw, db) of the Last layer
        dz = Cost_der(in_a[-1], sample[1])* P_sigmoid(out_z[-1]) #(dz is the delta value)
        dw[-1] = np.append( in_a[-2], 1).transpose()*dz # (1,N) * (N,1)
        #db[-1] = dz #(N,1)
        #2. Get iterative, # for i the iterative (from last layerN)
        #a0 --- (z0)a1 --- (z1)a2 ---(z2) : a3 --- C
        #    w0         w1         w2
        for i in range(N-2, -1,-1) : 
            dz = np.dot(np.transpose(self.weights[i+1][:,:-1]),dz) * P_sigmoid(out_z[i])
            dw[i] = np.append(in_a[i], 1).transpose()*dz # (1,N) * (N,1)
        #    db[i] = dz #(N,1)
        #3. 
        return (dw)#, db)
    
    def backpropagation_batch(self, batch):
        #Repackage input nodes of each samples into 784 ny N matrix
        inp = batch[0][0]
        outp = batch[0][1]
        N_batch = len(batch)
        for i in range(1, N_batch):
            inp = np.hstack((inp, batch[i][0]))
            outp = np.hstack((outp, batch[i][1]))
        in_a = [inp]
        out_z = []
        ina = copy.deepcopy(in_a[0])
        for w in self.weights:
            z = np.dot(w, np.vstack((ina, np.ones((1,N_batch)) )))
            ina = sigmoid(z)
            out_z.append(z)
            in_a.append(ina)
        dw = [np.zeros_like(w) for w in self.weights]
        
        N_z = len(out_z)
        dz = Cost_der(in_a[-1], outp)* P_sigmoid(out_z[-1])
        #
        dim_a=in_a[-2].shape
        dim_z=dz.shape
        dw[-1] = np.sum( np.vstack([in_a[-2],np.ones((1,N_batch))]).T.reshape(N_batch,1 , dim_a[0]+1)*dz.T.reshape(N_batch,dim_z[0] ,1)
                , axis=0)
        '''
        for i in range(N_batch):
            dw[-1] += np.vstack( [in_a[-2][:,[i]],np.ones((1,1))]).transpose()*dz[:,[i]]
        '''
        for i in range(N_z-2, -1, -1):
            dz = np.dot(np.transpose(self.weights[i+1][:,:-1]),dz) * P_sigmoid(out_z[i])
            dim_a=in_a[i].shape
            dim_z=dz.shape

            dw[i] = np.sum( np.vstack([in_a[i],np.ones((1,N_batch))]).T.reshape(N_batch,1 , dim_a[0]+1)*dz.T.reshape(N_batch,dim_z[0] ,1)
                , axis=0)
        '''
        for j in range(N_batch):
                dw[i] += np.vstack( [in_a[i][:,[j]],np.ones((1,1))]).transpose()*dz[:,[j]]    
        '''
        return (dw)
        
 
    def feedforward(self, X):
        a=[]
        
        for in_a, label in X:
            ina = copy.deepcopy(in_a)
            #weights and biases are yet seperated.
            for w in self.weights:
                ina = sigmoid(np.dot(w, np.vstack( (ina,[1]) ) ) )
            a.append(np.argmax(ina) == label)
        return (np.array(a))


    def evaluate(self, test):
        #Return Onehot Encoding | Assume x:0, y:1
        return (np.sum( self.feedforward(test))/len(test))

#%%
import timeit
if __name__ == "__main__":
    training, validation, test = inda.load_data_wrapper()
    #(784, 10, 10)
    net_dim = [784, 10, 10]
    epochs = 10
    nn = Network(net_dim)
    start = timeit.default_timer()
    weights = nn.stochastic_gradient_descent(training, 1, test=test)
    weights = nn.stochastic_gradient_descent(training, epochs, test=test, Ws=weights)
    end = timeit.default_timer()
    print (start - end)
