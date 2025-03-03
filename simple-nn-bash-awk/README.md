# Bash Script: Neural Network with AWK

This Bash script implements a simple neural network using AWK for data processing and training. Below is a detailed breakdown of its components, including steps, formulas, and code flow.

## 1. Data Loading and Vocabulary Building

The script begins by building vocabularies for words and characters from a dataset (`dataset.csv`).

### Word and Character Counting:

It reads each line from the dataset, splits it into words (`w1`, `w2`, `w3`) and a target string (`target`). It then counts the occurrences of each word and character using associative arrays `word_counts` and `char_counts`.

### Vocabulary Files:

Unique words and characters are extracted and saved into files (`model_words.txt` and `model_chars.txt`).

## 2. Training Core

The core training is handled by an AWK script.

### Initialization:

The script initializes input size (`INPUT_SIZE`), output size (`OUTPUT_SIZE`), and weight matrices (`W1` and `W2`) with small random values.

### Training Loop:

It iterates through epochs, performing the following steps:

#### Data Preparation:

Each input word is converted to a one-hot encoded vector. The target string is also converted into a one-hot encoded vector.

#### Forward Pass:

##### Hidden Layer Calculation:

The input vector is multiplied by the first weight matrix (`W1`), and a sigmoid activation function is applied:

hidden[j] = 1 / (1 + e^(-Σ(input[i] * W1[i,j])))


##### Output Layer Calculation:

The hidden layer output is multiplied by the second weight matrix (`W2`):

output[k] = Σ(hidden[j] * W2[j,k])


##### Softmax Function:

The output is passed through a softmax function to obtain probabilities:

probs[k] = e^(output[k]) / Σ e^(output[k])


#### Error Calculation:

The error is computed as the mean squared error between the predicted probabilities and the target values:

error = (1 / OUTPUT_SIZE) * Σ (probs[k] - target[k])^2


#### Backpropagation:

##### Output Layer Delta:

δ3[k] = probs[k] - target[k]


##### Hidden Layer Delta:

δ2[j] = (Σ(δ3[k] * W2[j,k])) * hidden[j] * (1 - hidden[j])


#### Weight Updates:

##### Second Layer Weights:

W2[j,k] = W2[j,k] - LEARNING_RATE * δ3[k] * hidden[j]


##### First Layer Weights:

W1[i,j] = W1[i,j] - LEARNING_RATE * δ2[j] * input[i]


#### Metrics Calculation:

After each epoch, the script calculates the error rate and accuracy, printing them to the console.

#### Model Saving:

The weight matrices (`W1` and `W2`) are saved to files after each epoch.

## 3. Interactive Prediction

After training, the script enters a prediction mode, allowing users to input three space-separated words.

### Input Processing:

The input words are converted to a one-hot encoded vector using the previously built vocabulary.

### Forward Pass:

The script performs a forward pass through the network using the trained weights to compute the hidden and output layers.

### Prediction:

The output probabilities are processed to determine the most likely characters for each position, forming the predicted string, which is then displayed to the user.

## Neural Network Diagram

Below is a diagram representing the neural network architecture used in the script:

Figure: A simple neural network with an input layer, one hidden layer, and an output layer. Each node represents a neuron, and each connection represents a weight.

This diagram illustrates the flow of data from the input layer through the hidden laye
