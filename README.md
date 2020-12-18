# MLisML - Machine Learning is Matrix muLtiplication

## Purpose
This repository is exclusively used for the realization of the project for the Machine Learning exam @UniPi for the academic year 2020/21.

## Usage
Just like Keras, but a lot uglier.

Example:
```
from neural_networks import NeuralNetwork
from layers import InputLayer, DenseLayer, OutputLayer
from dataset import MLCupDataset

# Create a MLCupDataset object containing MLCup patterns and targets
data = MLCupDataset()

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
```


## How to contribute
Don't.

## Contributors
Team DeepMai:

[Dario Salvati](d.salvati2@studenti.unipi.it)

[Andrea Zuppolini](a.zuppoolini@studenti.unipi.it)
