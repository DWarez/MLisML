'''

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
'''

import numpy as np
from functions._activation_functions import activation_functions
from functions._init_functions import init_functions


class Layer:
    '''
        Base class for the layers

        Attributes
        ----------

        units (int): 
                Non negative integer indicating the number of neurons in the layer
        activation (string): 
            Activation function used in the layer
        weight_initializer (string): 
            Initializer function used for generating strating weights
        bias_initializer (string):
            Initializer function used for generating bias
        _weights (list):
            Matrix containing layer's weights
        _old_weights (list):
            Matrix containing previous wieghts, used for momentum
        _fanin (list):
            Fanin of the layer
        _net (list):
            Net of the layer
        _deltas (list):
            Deltas of the layer, computed during backpropagation
        type (string):
            Type of the layer

    '''

    def __init__(self, units, fanin, activation, initializer):
        '''
            Parameters
            ----------

            units (int): 
                Non negative integer indicating the number of neurons in the layer
            activation (string): 
                Activation function used in the layer
            initializer (string): 
                Initializer function used for generating strating weights and bias
        '''
        if isinstance(units, int) and units > 0:
            self.units = units
        else:
            raise ValueError("Number of units must be a non negative integer, please try a different value.")
        
        if activation in activation_functions:
            self.activation = activation
        else:
            raise ValueError("Activation function is not implemented or doesn't exist, please try a different one.")

        if initializer in init_functions:
            self.weight_initializer = initializer
        else:
            raise ValueError("Initializer function is not implemented or doesn't exist, please try a different one.")
        
        self.bias_initializer = initializer

        self._weights = []
        self._old_weights = []
        self._fanin = []
        self._net = []
        self._deltas = []
        self.type = ""

        '''
            On each row we get the weights for the neuron, however the first position is actually the bias of the neuron
        '''


    '''
        Defining iteration methods for the Layer object
    '''
    def __str__(self):
        '''
            Defining __str__ function for pretty print
        '''
        return "Layer type: {}\nNumber of units: {}\nActivation function: {}\nWeights initializer function: {}\nBias initializer function: {}\n\n".format(self.type, self.units, self.activation, self.weight_initializer, self.bias_initializer)


class InputLayer(Layer):
    '''
        Standard class for input layer
    '''
    def __init__(self, units, fanin = 1, activation = "identity", initializer = "normal"):
        '''
            Parameters
            ----------

            units (int): 
                Non negative integer indicating the number of neurons in the layer
            activation (string): 
                Activation function used in the layer
            initializer (string): 
                Initializer function used for generating strating weights and bias
        '''
        super().__init__(units, fanin, activation, initializer)
        self.type = "Input"

        for _ in range(units):
            if initializer == "normal":
                self._weights.append(np.random.randn(units + 1))
            elif initializer == "uniform":
                self._weights.append(np.random.rand(units + 1))
            elif initializer == "he":
                self._weights.append(np.random.randn(units + 1) * np.sqrt(2/fanin))
            else:
                raise ValueError("Initializer not found")
        
        self._old_weights = np.zeros((np.array(self._weights).shape[0], np.array(self._weights).shape[1] -1))


class OutputLayer(Layer):
    '''
        Standard class for the output layer
    '''
    def __init__(self, units, fanin, activation = "identity", initializer = "normal"):
        '''
            Parameters
            ----------

            units (int): 
                Non negative integer indicating the number of neurons in the layer
            activation (string): 
                Activation function used in the layer
            initializer (string): 
                Initializer function used for generating strating weights and bias
        '''
        super().__init__(units, fanin, activation, initializer)
        self.type = "Output"

        for _ in range(units):
            if initializer == "normal":
                self._weights.append(np.random.randn(fanin + 1))
            elif initializer == "uniform":
                self._weights.append(np.random.rand(fanin + 1))
            elif initializer == "he":
                self._weights.append(np.random.randn(fanin + 1) * np.sqrt(2/fanin))
            else:
                raise ValueError("Initializer not found")
        
        self._old_weights = np.zeros((np.array(self._weights).shape[0], np.array(self._weights).shape[1] -1))


class DenseLayer(Layer):
    '''
        Standard class for the dense layer
    '''
    def __init__(self, units, fanin, activation = "sigmoid", initializer = "normal"):
        '''
            Parameters
            ----------

            units (int): 
                Non negative integer indicating the number of neurons in the layer
            activation (string): 
                Activation function used in the layer
            initializer (string): 
                Initializer function used for generating strating weights and bias
        '''
        super().__init__(units, fanin, activation, initializer)
        self.type = "Dense"

        for _ in range(units):
            if initializer == "normal":
                self._weights.append(np.random.randn(fanin + 1))
            elif initializer == "uniform":
                self._weights.append(np.random.rand(fanin + 1))
            elif initializer == "he":
                self._weights.append(np.random.randn(fanin + 1) * np.sqrt(2/fanin))
            else:
                raise ValueError("Initializer not found")
        
        self._old_weights = np.zeros((np.array(self._weights).shape[0], np.array(self._weights).shape[1] -1))


layer_types = ["Input", "Output", "Dense"]