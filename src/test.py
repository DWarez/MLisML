from neural_networks import NeuralNetwork
from layers import InputLayer, DenseLayer, OutputLayer
import dataset as ds

# Create a MLCupDataset object containing MLCup patterns and targets
data = ds.MLCupDataset()

# The framework also contains MONKS
# data = ds.MonksDataset()

# Initialize the model and add the layers
my_model = NeuralNetwork()
my_model.add(InputLayer(10))
my_model.add(DenseLayer(15, fanin = 10, activation="sigmoid"))
my_model.add(OutputLayer(2, fanin = 15))

# Setup the model's hyperparameters and training parameters
my_model.compile(857, 600, 0.1/1524, None, 0.001, 0.01, "mean_squared_error")

# Fit the model using the previously defined dataset
my_model.fit(data.train_data_patterns, data.train_data_targets)