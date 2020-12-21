import numpy as np

import dataset as ds
from neural_networks import NeuralNetwork
from layers import InputLayer, OutputLayer, DenseLayer
from functions._init_functions import init_functions
from functions._activation_functions import activation_functions, activation_functions_derivatives
from functions._loss_functions import loss_functions
import plot as plt


data = ds.MLCupDataset()

my_model = NeuralNetwork()
my_model.add(InputLayer(10))
my_model.add(DenseLayer(100, fanin = 10))
my_model.add(DenseLayer(100, fanin = 100))
my_model.add(OutputLayer(2, fanin = 100))

my_model.compile(857, 400, 0.01/857, None, 0.001, 0.3, "mean_squared_error")

loss = my_model.fit(data.train_data_patterns, data.train_data_targets)

print(loss[-1])
print("evaluation: {}".format(my_model.evaluate(data.model_assessment_patterns, data.model_assessment_targets)))
plt.plot_loss(loss)