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

'''
    Scalar Functions
'''

def scalar_uniform_init():
    ''' 
        Uniform distribution initialization.
        :return: value from the uniform distribution
    '''
    return np.random.rand(1)[0]


def scalar_normal_init():
    '''
        Normal distribution initialization.
        :return: value from the normal distribution
    '''
    return np.random.randn(1)[0]

def scalar_zero_init():
    return 0

'''
    Defining the dictionary for scalar weight initialization functions
'''
scalar_init_functions = {
    "uniform": scalar_uniform_init,
    "normal": scalar_normal_init,
    "zero": scalar_zero_init
}



'''
    Vectorial Functions
'''

def uniform_init(x):
    '''
        Uniform distribution initialization.
        :param x: narray from which the shape is taken
        :return: narray of the same shape of the parameter containing values from the uniform distribution
    '''
    return np.random.rand(x)


def normal_init(x):
    '''
        Normal distribution initialization.
        :param x: narray from which the shape is taken
        :return: narray of the same shape of the parameter containing values from the normal distribution
    '''
    return np.random.randn(x)


def he_init(x, fanin_size):
    '''
        He initialization, computed as: normal_init(x) * sqrt(2/fain_size).
        :param x: narray from which the shape is taken
        :return: narray of the same shape of the parameter containing values from the normal distribution
    '''
    return np.random.randn(*(x.shape)) * np.sqrt(2/fanin_size)


def zero_init(x):
    return np.zeros(x.shape)

'''
    Defining the dictionary for vectorial weight initialization functions
'''
init_functions = {
    "uniform": uniform_init,
    "normal": normal_init,
    "he": he_init,
    "zero": zero_init
}