# MNIST-Digit-Classification

This project contains a modified implementation of a neural network designed to classify handwritten digits from the MNIST dataset. The network uses stochastic gradient descent for training, and the code has been improved to provide more detailed feedback on performance during training and testing.

## Original Code

The original code for this project can be found at [Deep Learning Python - GitHub](https://github.com/MichalDanielDobrzanski/DeepLearningPython). Our modifications are based on this implementation, with adjustments to enhance training and testing outputs.

## Project Requirements

For this project, the task was to:

1. Modify the existing `network.py` and `test.py` example codes to train a feed-forward neural network with three layers:
   - Input layer of 784 nodes
   - One hidden layer of 30 nodes
   - Output layer of 10 nodes (using sigmoid neurons)
2. Use the stochastic gradient descent algorithm with a mini-batch size of 10 and a learning rate of 3.0.
3. Train the network on 50,000 images from the MNIST training dataset and evaluate it on 10,000 test images.
4. Revise the code to show the total number of correctly classified and incorrectly classified images for each digit class (0-9) after every epoch.
5. After the last epoch, display the incorrectly classified images, showing the correct and predicted labels for each sample.

The modified implementation should:

- Track and display classification results for 20 epochs.
- Save examples of incorrect classifications to aid in performance analysis.

## Files

- **`network.py`**: The main neural network training script, modified for improved learning rate, batch size, and training feedback.
- **`test_modified_semester_project.py`**: Custom testing script that evaluates the trained network and displays detailed results, including examples of incorrectly classified digits.

## Features

- Three-layer neural network (input, hidden, output) with sigmoid activation functions
- Enhanced training output showing epoch-wise accuracy
- Detailed testing results for each digit class (0-9)
- Visualization of misclassified images to help identify patterns and improve model performance

## Usage

1. Run `network.py` to train the neural network on the MNIST dataset.
2. Use `test_modified_semester_project.py` to evaluate the network on test data and review the results.
