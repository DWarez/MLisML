import numpy as np

import dataset as ds
from neural_networks import NeuralNetwork
from layers import InputLayer, OutputLayer, DenseLayer
from functions._init_functions import init_functions
from functions._activation_functions import activation_functions, activation_functions_derivatives
from functions._loss_functions import loss_functions
import plot as plt


data = ds.MLCupDataset()

data = ds.MLCupDataset()

model = NeuralNetwork()
model.add(InputLayer(10))
model.add(DenseLayer(50, fanin=10, activation="sigmoid"))
model.add(DenseLayer(30, fanin=50, activation="sigmoid"))
model.add(OutputLayer(2, fanin=30))

# configuration 322, line 324
model.compile(1142, 600, 0.03/1142, None, 0.000008, 0.3, "mean_squared_error")

loss = model.fit(data.train_data_patterns, data.train_data_targets)

i = 0
final_loss = []
while i < len(loss):
    final_loss.append((loss[i] + loss[i+1])/2)
    i += 2

print(final_loss[-1])
model._blind_test(data.test_data_patterns)