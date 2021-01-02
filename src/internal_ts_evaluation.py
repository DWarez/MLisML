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
model.add(DenseLayer(20, fanin=10, activation="sigmoid"))
model.add(OutputLayer(2, fanin=20))


model.compile(1143, 600, 0.01, None, 0.000008, 0.08, "mean_squared_error")

loss = model.fit(data.train_data_patterns, data.train_data_targets)
print(loss[-1])

ts_evaluation = model.evaluate(data.model_assessment_patterns, data.model_assessment_targets)
print(ts_evaluation)