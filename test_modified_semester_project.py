import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np

def display_incorrect_instances(incorrect_classifications):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))  

    for digit in range(10):  # Iterate through each digit from 0 - 9
        ax = axes[digit]
        info = incorrect_classifications.get(digit)
        if info and info['first_instance']:  # Check if there's an incorrect instance
            x, prediction = info['first_instance']
            image = x.reshape(28, 28)  
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title(f'True: {digit}, Pred: {prediction}')
        else:
            ax.text(0.5, 0.5, 'No errors', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off') # Axis is turned off for better visual appearance
    plt.tight_layout()
    plt.show()

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Initialize and train the network
net = network.Network([784, 30, 10])

# Network training variables
epochs = 20
mini_batch_size = 10
learning_rate = 3.0

# Train the network and collect incorrect classifications
incorrect_classifications = net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

# Display the first incorrect instances after the last epoch
display_incorrect_instances(incorrect_classifications)