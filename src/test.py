import numpy as np

import dataset as ds
from neural_networks import NeuralNetwork
from layers import InputLayer, OutputLayer, DenseLayer
from functions._init_functions import init_functions
from functions._activation_functions import activation_functions, activation_functions_derivatives
from functions._loss_functions import loss_functions
import plot as plt


data = ds.MLCupDataset()


k = 10

for i in range(k):
    (train, val) = data.kfolds(i)
    my_model = NeuralNetwork()
    my_model.add(InputLayer(10))
    my_model.add(DenseLayer(15, fanin = 10, activation="sigmoid"))
    my_model.add(OutputLayer(2, fanin = 15))
    my_model.compile(857, 600, 0.1/1524, None, 0.001, 0.01, "mean_squared_error")
    print("loss: {}".format(my_model.fit(train[0], train[1])[-1]))
    print("evaluation: {}".format(my_model.evaluate(val[0], val[1])))