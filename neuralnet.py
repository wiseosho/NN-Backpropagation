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
import _pickle
import gzip
import copy

# Third-party libraries
import numpy as np

#%%
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = _pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = [ [inputs, results] for inputs, results in zip(training_inputs, training_results) ] # zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = [ [ va_i, va_o   ] for va_i, va_o in zip(validation_inputs, va_d[1])]#zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = [ [te_i, te_o] for te_i, te_o in zip(test_inputs, te_d[1]) ]#zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
#%%
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def P_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def Cost_der(h, y):
    return (h-y)

#%%
class Network(object):
    
    def __init__(self, net_dim, lr=0.1, Ws=None, Bs=None): # net_dim = [#nodes layer, start from input nodes]
        if (Ws == None) or (Bs == None):
            self.weights = [np.random.randn(o, i) for i, o in zip(net_dim[:-1], net_dim[1:] )]
            self.biases = [np.random.randn(o, 1) for o in net_dim[1:] ]
        
        else:
            self.weights = Ws
            self.biases = Bs
        self.lr = lr
        #self.activations = np.zeros((Nlayer, ))
        #self.output = 
        
    def stochastic_gradient_descent(self,train, epochs, test=None, lr=0.5, Bsize = 200, Ws=None, Bs=None):
        if (Ws !=None and Bs != None) :
            self.weights = Ws
            self.biases = Bs
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
        return (self.weights, self.biases)

    def update_from_batch(self,batch):
        #batch is composed of x, y of input node, labels
        N = len(batch)
        #make placeholders for delta_parameters
        accum_delta_w = [np.zeros_like(w) for w in self.weights]
        accum_delta_b = [np.zeros_like(b) for b in self.biases]
        
        #calculate backpropagated derivatives of Cost function w.r.t weights and biases
        # This 
        for sample in batch: # Suppose the sample consists of [input, label], where shape(input) = n,1/
            del_w, del_b = self.backpropagation(sample) # accum_delta_w, accum_delta_b #
        
            #add calculated sample into (accumulate)
            accum_delta_w = [acc_del_w + del_w for acc_del_w, del_w in zip(accum_delta_w,del_w )]
            accum_delta_b = [acc_del_b + del_b for acc_del_b, del_b in zip(accum_delta_b, del_b)]

        # weight correction using accumulated delta with learning rate(lr)
        self.weights = [w - (self.lr/N)*del_w for w, del_w in zip(self.weights, accum_delta_w)]
        self.biases = [b - (self.lr/N)*del_b for b, del_b in zip(self.biases, accum_delta_b)]
    def backpropagation(self, sample):
        #Suppose sample is One case with N feasures in shape of matrix(N,1)
        #activations and sum values of each layer should be counted and listed in proccess of feedforward 
        #
        #Layer_val = np.zeros(1,1)
        #Storing intermediate output values will be appended from existing list
        in_a = [sample[0]] #first value will be the input features
        out_z = []
        ina = copy.deepcopy(sample[0])
        #Feed_forward and save each layer output
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, ina)+b
            ina = sigmoid(z)
            out_z.append(z)
            in_a.append(ina)
        #Backpropagation using parameters
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        N = len(out_z)
        #1. Get Cost derivative values(dz, dw, db) of the Last layer
        dz = Cost_der(in_a[-1], sample[1])* P_sigmoid(out_z[-1]) #(dz is the delta value)
        dw[-1] = in_a[-2].transpose()*dz # (1,N) * (N,1)
        db[-1] = dz #(N,1)
        #2. Get iterative, # for i the iterative (from last layerN)
        #a0 --- (z0)a1 --- (z1)a2 ---(z2) : a3 --- C
        #    w0         w1         w2
        for i in range(N-2, -1,-1) : 
            dz = np.dot(np.transpose(self.weights[i+1]),dz) * P_sigmoid(out_z[i])
            dw[i] = in_a[i].transpose()*dz # (1,N) * (N,1)
            db[i] = dz #(N,1)
        #3. 
        return (dw, db)

    def feedforward(self, X):
        a=[]
        
        for in_a, label in X:
            ina = copy.deepcopy(in_a)
            #weights and biases are yet seperated.
            for w, b in zip(self.weights, self.biases):
                ina = sigmoid(np.dot(w, ina) + b)
            a.append(np.argmax(ina) == label)
        return (np.array(a))


    def evaluate(self, test):
        #Return Onehot Encoding | Assume x:0, y:1
        return (np.sum( self.feedforward(test))/len(test))

#%%
if __name__ == "__main__":
    training, validation, test = load_data_wrapper()
    #(784, 10, 10)
    net_dim = [784, 10, 10]
    epochs = 1
    nn = Network(net_dim)
    weights, biases = nn.stochastic_gradient_descent(training, 1, test=test)
    weights, biases = nn.stochastic_gradient_descent(training, epochs, test=test, Ws=weights, Bs=biases)


    
