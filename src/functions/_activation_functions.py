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

'''
    This module contains a set of activation functions and their respective 
    first order derivatives. It's then defined a dictionary thant contains all the functions.
'''

import numpy as np


def identity(x):
    '''
        Identity function f(x) = x
    '''
    x = np.clip(x, -700, 700)
    return np.array(x)

def binary_step(x):
    '''
        Binary step function f(x) = 1 for x >= 0, f(x) = 0 for x < 0
    '''
    if x >= 0:
        return 1
    else:
        return 0

def sigmoid(x):
    '''
        Sigmoid function f(x) = 1/(1 + e^(-x))
    '''
    x = np.clip(x, -20, 20)
    return 1.0/(1 + np.exp(-x))

def tanh(x):
    '''
        Tanh function f(x) = (e^2x - 1)/(e^2x + 1)
    '''
    return np.tanh(x)

def relu(x):
    '''
        REctified Linear Unit function f(x) = max(0, x)
    '''
    #x = np.clip(x, -20, 20)
    return np.maximum(x, 0)

def softmax(x):
    '''
        Softmax function f(x) = (e^x)/(sum(e^x))
    '''
    s = np.sum(np.exp(x))
    return (np.exp(x))/s


'''
    Defining the functions dictionary
'''

activation_functions = {
    "identity": identity,
    "binary_step": binary_step,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "softmax": softmax
}



'''
    First order derivatives
'''

def identity_derivative(x):
    '''
        First order derivative of the identity function f'(x) = 1
    '''
    return np.zeros(x.shape) + 1

def binary_step_derivative(x):
    '''
        First order derivative of the binary step function f'(x) = 0 for x ≠ 0
    '''
    return np.zeros(x.shape) + 1

def sigmoid_derivative(x):
    '''
        First order derivative of the sigmoid function f'(x) = f(x)(1 - f(x))
    '''
    x = np.clip(x, -20, 20)
    return np.exp(-x) / ((np.exp(-x) + 1)**2)

def tanh_derivative(x):
    '''
        First order derivative of the tanh function f'(x) = 1 - f^2(x)
    '''
    #x = np.clip(x, -300, 300)
    return (1 - np.power(np.tanh(x), 2))

def relu_derivative(x):
    '''
        First order derivative of the ReLU function f'(x) = 1 for x > 0, f'(x) = 0 for x <= 0
    '''
    x[x>0] = 1
    x[x<=0] = 0
    return x
    

def softmax_derivative(x):
    '''
        First order derivative of the softmax function D_jS_i = S_i(d_ij + S_i) where d_ij = 1 if i = j, else 0
    '''
    s = softmax(x)
    s = s.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)




'''
    Defining the functions first order derivative dictionary
'''

activation_functions_derivatives = {
    "identity": identity_derivative,
    "binary_step": binary_step_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "relu": relu_derivative,
    "softmax": softmax_derivative
}