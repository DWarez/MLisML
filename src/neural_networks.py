"""

    @author Dario Salvati <d.salvati2@studenti.unipi.it>
    @author Andrea Zuppolini <a.zuppolini@studenti.unipi.it>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import math
import random
import progressbar

from functions._activation_functions import activation_functions, activation_functions_derivatives
from functions._loss_functions import loss_functions, loss_functions_derivatives
from functions._init_functions import init_functions
from layers import Layer, layer_types

np.seterr(all="raise")

class NeuralNetwork:
    """
        Implementation of a basic feedforward Neural Network

        Attributes
        ----------

        model (list):
            List containing all the layers of the model. You can add layers using the add method.
        nlayers (int):
            Number of layers of the model
        batch_size (int):
            Size of the batch used for training
        epochs (int):
            Number of epochs used for training
        learning_rate (float): 
            Learning rate value
        optimizer (string): 
            Optimizer used
        regularization (float): 
            Coefficient used for the regularization term
        momentum (float): 
            Coefficient used for the momentum term
        loss (string): 
            Loss function used during the training of the model 
    """

    def __init__(self):
        self._model = []
        self.nlayers = 0


    def compile(self, batch_size, epochs, learning_rate, optimizer, regularization, momentum, loss):
        """
            Parameters
            ----------

            batch_size (int):
                Size of the batch used for training

            epochs (int):
                Number of epochs used for training

            learning_rate (float): 
                Learning rate value

            optimizer (string): 
                Optimizer used

            regularization (float): 
                Coefficient used for the regularization term

            momentum (float): 
                Coefficient used for the momentum term
                
            loss (string): 
                Loss function used during the training of the model
        """
        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise ValueError("Batch size must be an integer")
        
        if isinstance(epochs, int):
            self.epochs = epochs
        else:
            raise ValueError("Number of epochs must be an integer")     

        self.learning_rate = learning_rate
        self.optimizer = optimizer  #check optimizer
        self.regularization = regularization
        self.momentum = momentum

        if loss in loss_functions:
            self.loss = loss
        else:
            raise ValueError("Loss function is not implemented or doesn't exist.")
        
        
        """
            Grabbing actual functions from imports
        """
        #same for optimizer
        self._loss_function = loss_functions[self.loss]


    def get_params(self):
        """
            Getter of the class

            Returns
            -------
            return: 
                Dictionary cointaining all the values of the attributes of the istance.
        """
        return {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "regularization": self.regularization,
            "momentum": self.momentum,
            "max_iterations": self.epochs,
            "batch_size": self.batch_size,
            "loss": self.loss,
        }


    def add(self, layer):
        """
            Method that allows the insertion of a specified layer into the model instance.

            Parameters
            ----------

            layer (Layer): an object of the Layer class which will be added to the model
        """
        if not isinstance(layer, Layer):
            raise ValueError("The parameter is not a layer.")

        if layer.type not in layer_types:
            raise ValueError("The layer type is not yet implemented or doensn't exist")
        
        if self.nlayers == 0 and layer.type != "Input":
            raise ValueError("The first layer of the model must be an InputLayer.")
        
        if self.nlayers > 0 and self._model[-1].type == "Output":
            raise ValueError("The model has an output layer, hence you cannot add more layers.")

        self._model.append(layer)

        self.nlayers += 1

    
    def _feedforward(self, patterns):
        """
            Method that computes the neurons values for a given input.

            Parameters
            ----------

            patterns (list): 
                List of input patterns fed to the model
        """
        activation_matrix = []  # matrix containing the output values of the layer
        for i in range(len(self._model)):
            if i == 0:
                self._model[i]._fanin = patterns.T      # setting the fanin of the InputLayer as the transpose of the patterns
                patterns = np.insert(patterns.T, 0, 1, axis=0)      # prepend 1 into each array of patterns to compy with matrix multiplication
                self._model[i]._net = np.dot(self._model[i]._weights, patterns)     # computing the net
                activation_matrix = activation_functions[self._model[i].activation](self._model[i]._net)  # computing the activation of the net  
            else:
                # The fanin of each non-Input Layer is the activation_matrix at the previous step
                self._model[i]._fanin = activation_matrix
                activation_matrix = np.insert(activation_matrix, 0, 1, axis=0)
                self._model[i]._net = np.dot(self._model[i]._weights, activation_matrix)
                activation_matrix = activation_functions[self._model[i].activation](self._model[i]._net)
        
        return activation_matrix


    def _backpropagation(self, patterns, targets):
        """
            Method for backpropagation.

            Returns
            -------

            loss (list):
                Loss of the model before training of the current minibatch.
        """

        """
            Dimentions spreadsheet

            outputs = #neurons of OutputLayer = dimensions of output X #patterns
            targets = #patterns X #dimensions of output = #neurons of OutputLayer
            gradients = #neurons of OutputLayer = dimensions of output X #patterns
            nets = #neurons of the layer X #patterns
            activation_derivatives = #neurons of the layer X #patterns
            deltas = #neurons of the layer X #patterns
            fanin = #neurons fanin X #patterns
        """
        """
            Compute outputs for each pattern in the batch
        """

        '''
            Preprocessing used for managing Nesterov's momentum.
            Save the real weights, compute weights tilde, compute gradient and then get back real weights
        '''
        real_weights = []
        for i in range(len(self._model)):
            real_weights.append(self._model[i]._weights)
            biases = np.array(self._model[i]._weights)[:, 0]
            self._model[i]._weights = np.delete(self._model[i]._weights, 0, 1)
            self._model[i]._weights = np.add(self._model[i]._weights, np.subtract(self._model[i]._weights, self._model[i]._old_weights) * self.momentum)
            self._model[i]._weights = np.insert(self._model[i]._weights, 0, biases, axis=1)

        outputs = self._feedforward(patterns)

        for i in range(len(self._model)):
            self._model[i]._weights = real_weights[i]

        loss = loss_functions[self.loss](outputs, targets.T)

        """
            Compute updates for output layer
        """
        gradients = loss_functions_derivatives[self.loss](outputs, targets.T)     # compute the gratients matrix (outputs - targets)
        #gradients = np.clip(gradients, -0.5, 0.5)
        activation_derivatives = activation_functions_derivatives[self._model[-1].activation](self._model[-1]._net) # matrix of derivatives of the nets
        #activation_derivatives = np.clip(activation_derivatives, -0.5, 0.5)
        self._model[-1]._deltas = np.multiply(gradients, activation_derivatives)    # compute deltas for the output layer
        biases = np.array(self._model[-1]._weights)[:, 0]       # save biases of layer
        self._model[-1]._weights = np.delete(self._model[-1]._weights, 0, 1)    # remove biases from weights matrix
        """
            First save current weights in tmp.
            Then compute weights update: W = W - ((DELTAS X FANIN.T) * lr + alpha * (W - Wold) - lambda * W)
            Then assign to Wold the value of tmp.
        """
        tmp = np.array(self._model[-1]._weights)
        self._model[-1]._weights = np.subtract(self._model[-1]._weights, np.subtract(np.add(np.dot(self._model[-1]._deltas, \
            np.transpose(self._model[-1]._fanin)) * self.learning_rate, self.momentum * np.subtract(self._model[-1]._weights, self._model[-1]._old_weights)), \
                self.regularization * self._model[-1]._weights))
        self._model[-1]._old_weights = tmp
        self._model[-1]._weights = np.insert(self._model[-1]._weights, 0, biases, axis=1)   # add back the bias vector

        """
            Compute updates for hidden layers
        """
        for l in range(len(self._model) - 2, -1, -1):
            self._model[l]._deltas = np.multiply((np.dot(np.transpose(self._model[l+1]._weights)[1:, :], self._model[l+1]._deltas)), \
                activation_functions_derivatives[self._model[l].activation](self._model[l]._net))   # compute deltas matrix for the hidden layer
            biases = np.array(self._model[l]._weights)[:, 0]    # save biases vector
            self._model[l]._weights = np.delete(self._model[l]._weights, 0, 1)  # remove bias vector from weights
            """
                First save current weights in tmp.
                Then compute weights update: W = W - ((DELTAS X FANIN.T) * lr + alpha * (W - Wold) - lambda * W)
                Then assign to Wold the value of tmp.
            """
            tmp = np.array(self._model[l]._weights)
            self._model[l]._weights = np.subtract(self._model[l]._weights, np.subtract(np.add(np.dot(self._model[l]._deltas, \
                np.transpose(self._model[l]._fanin)) * self.learning_rate, self.momentum * np.subtract(self._model[l]._weights, self._model[l]._old_weights)), \
                    self.regularization * self._model[l]._weights))
            self._model[l]._old_weights = tmp
            self._model[l]._weights = np.insert(self._model[l]._weights, 0, biases, axis=1) # add back the bias vector

        return loss
        

    def fit(self, patterns, targets):
        """
            Implementation of the fit method for the NeuralNetwork

            Parameters
            ----------
            
            patterns (list):
                List containing the patterns used for training
            targets (list):
                List containing the labels of the patterns used for training

            Returns
            -------
            loss (list):
                List containing the losses after every epoch of training
        """
        loss = []
        i = 0

        for _ in range(self.epochs):
            for i in range(math.floor(len(patterns)/self.batch_size)):
                loss.append(self._backpropagation(patterns[i * self.batch_size : (i + 1) * self.batch_size], targets[i * self.batch_size : (i + 1) * self.batch_size]))
            
            if len(patterns) > (i + 1) * self.batch_size:
                loss.append(self._backpropagation(patterns[(i + 1) * self.batch_size:], targets[(i + 1) * self.batch_size:]))
        
        i = 0
        l = []
        while i < len(loss):
            l.append((loss[i] + loss[i+1])/2)
            i += 2
        return l


    def evaluate(self, patterns, targets):
        """
            Method used to evaluate the accuracy of the trained model
        """
        if len(patterns) != len(targets):
            raise ValueError("The number of patterns and targets must be equal.")

        return loss_functions["mean_euclidean_error"](self._feedforward(patterns), targets.T)
    
    
    def __str__(self):
        """
            Defining __str__ function for pretty print
        """
        temp = ""
        count = 0
        for layer in self._model:
            temp += "Layer n. " + str(count) + "\n" + str(layer)
            count += 1
        return temp