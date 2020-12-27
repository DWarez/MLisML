import numpy as np

import dataset as ds
from neural_networks import NeuralNetwork
from layers import InputLayer, OutputLayer, DenseLayer
import plot as plt

data = ds.MonksDataset()

my_model = NeuralNetwork()
my_model.add(InputLayer(17))
my_model.add(DenseLayer(20, fanin = 17, activation="sigmoid"))
my_model.add(OutputLayer(1, fanin = 20, activation="sigmoid"))

my_model.compile(124, 600, 0.07, None, 0, 0, "mean_squared_error")

loss = my_model.fit(data.train_data_patterns, data.train_data_targets)
accuracy = my_model.evaluate_monks(data.test_data_patterns, data.test_data_targets)

print(loss[-1])
plt.plot_loss(loss)
print(accuracy)