import numpy as np

import dataset as ds
from neural_networks import NeuralNetwork
from layers import InputLayer, OutputLayer, DenseLayer
import matplotlib.pyplot as plt

data = ds.MonksDataset()

my_model = NeuralNetwork()
my_model.add(InputLayer(17))
my_model.add(DenseLayer(10, fanin = 17, activation="sigmoid"))
my_model.add(OutputLayer(1, fanin = 10, activation="sigmoid"))

my_model.compile(122, 600, 0.08, None, 0, 0, "mean_squared_error")


(loss, test_loss, accuracy, test_accuracy) = my_model.fit_monks(data.train_data_patterns, data.train_data_targets, data.test_data_patterns, data.test_data_targets)

print("Loss: {}".format(loss[-1]))
print("Test Loss: {}".format(test_loss[-1]))

print("Accuracy: {}".format(accuracy[-1]))
print("Test accuracy: {}".format(test_accuracy[-1]))

plot1 = plt.figure(1)
plt.plot(loss)
plt.plot(test_loss, "--")

plot2 = plt.figure(2)
plt.plot(accuracy)
plt.plot(test_accuracy, "--")

plt.show()
