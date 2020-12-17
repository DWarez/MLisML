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
    This module contains a set of loss functions and their respective 
    first order derivatives. The funcions are vectorized to be used with 
    numpy arrays and it's defined a structure thant contains all the vectorized functions.
'''

import math
import numpy as np

'''
    Loss functions
'''

def cross_entropy(predicted_values, true_values, epsilon = 1e-12):
    '''
        Cross entropy loss function
    '''
    predicted_values = np.clip(predicted_values, epsilon, 1. - epsilon)
    return -np.sum(true_values * np.log(predicted_values + 1e-9))/predicted_values.shape[0]


def mean_euclidean_error(predicted_values, true_values):
    '''
        Mean Absolute Error loss function (L1)
    '''
    if predicted_values.shape != true_values.shape:
        raise Exception("The number of predictions and the number of targets must be the same")

    return np.mean(np.sqrt(np.square(np.subtract(predicted_values, true_values))))


def mean_squared_error(predicted_values, true_values):
    '''
        Mean Squared Error loss function (L2)
    '''
    if predicted_values.shape != true_values.shape:
        raise ValueError("The dimentions of predicted values must be equal to the dimenentions of target values.")

    return np.mean(np.square(np.subtract(predicted_values, true_values)))



'''
    Loss functions first order derivatives.
    ToDo: add other derivatives
'''
def mean_squared_error_derivative(predicted_values, true_values):
    '''
        Mean Squared Error loss function (L2) derivative
    '''
    return np.subtract(predicted_values, true_values)


def cross_entropy_derivative(predicted_values, true_values):
    '''
        Derivative of cross entropy loss
    '''
    predicted_values = np.clip(predicted_values, 1e-5, 1)
    return (predicted_values - true_values) / (predicted_values - predicted_values**2)




'''
    Defining the loss functions dictionary
'''

loss_functions = {
    "cross_entropy": cross_entropy,
    "mean_euclidean_error": mean_euclidean_error,
    "mean_squared_error": mean_squared_error,
}

loss_functions_derivatives = {
    "cross_entropy": cross_entropy_derivative,
    "mean_squared_error": mean_squared_error_derivative,
}