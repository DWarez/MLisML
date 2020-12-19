from neural_networks import NeuralNetwork
from layers import InputLayer, OutputLayer, DenseLayer
from dataset import MLCupDataset
import os
import progressbar

DEFAULT_DIR = "/reports"
DEFAULT_NAME = "/model_selection.txt"


data = MLCupDataset()

learning_rates = [0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01]
epochs = [300, 600]
regularizations = [0.008, 0.005, 0.003, 0.001]
momentums = [0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01]

k = 10
size = 857
filepath = os.path.dirname(os.getcwd()) + DEFAULT_DIR + DEFAULT_NAME

fp = open(filepath, "w")

config = 0

for epoch in epochs:
    for lr in learning_rates:
        for reg in regularizations:
            for alpha in momentums:
                mean_loss = 0
                mean_validation = 0

                for i in range(k):
                    model = NeuralNetwork()
                    model.add(InputLayer(10))
                    model.add(DenseLayer(15, fanin = 10, activation="sigmoid"))
                    model.add(OutputLayer(2, fanin = 15))
                    model.compile(size, epoch, lr/size, None, reg, alpha, "mean_squared_error")
                    (train, val) = data.kfolds(index=i, k=k)
                    mean_loss = mean_loss + model.fit(train[0], train[1])[-1]
                    mean_validation = mean_validation + model.evaluate(val[0], val[1])

                fp.write("{}, {}, {}, {}, {}, {}, {}\n".format(config, epoch, lr, reg, alpha, mean_loss/k, mean_validation/k))

                config = config + 1

fp.close()