import matplotlib.pyplot as plt

def plot_loss(x):
    '''
        Function to plot the loss
    '''
    plt.plot(x)
    plt.title("Loss of the model")
    plt.ylabel("Loss")
    plt.show()