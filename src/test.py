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
my_model.add(InputLayer(10, initializer="he"))
my_model.add(DenseLayer(50, fanin = 10, initializer="he"))
my_model.add(DenseLayer(30, fanin = 50, initializer="he"))
my_model.add(OutputLayer(2, fanin = 30, initializer="he"))

my_model.compile(857, 600, 0.03/857, None, 0.00003, 0.5, "mean_squared_error")

loss = my_model.fit(data.train_data_patterns, data.train_data_targets)

i = 0
l = []
while i < len(loss):
    l.append((loss[i] + loss[i+1])/2)
    i += 2

print(l[-1])
print("evaluation: {}".format(my_model.evaluate(data.model_assessment_patterns, data.model_assessment_targets)))
plt.plot_loss(l)