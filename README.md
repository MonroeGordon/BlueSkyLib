# NeuralNetwork
A library for creating various neural network architectures. This library provides cpu and gpu processing support via the numpy, cupy, and scipy libraries.

## Multilayer Perceptron Neural Network
A multilayer perceptron neural network is a feed forward neural network with an input layer, hidden layer(s), amd an output layer. Each layer can have a differing number of neurons, but all neurons in a layer, typically, use the same activation function for processing their inputs. All neurons in a layer are connected to the next layer, with each connection having a weight. These weights are adjusted using backpropgation to get the network to align its outputs in the output layer with the true values pertaining to each input it is trained on. After training on a large number of example input/output pairs, a multilayer perceptron neural network can learn the pattern in the data, allowing it to predict the correct outputs for new inputs it was not trained on.

Included in this library:
- **mlp.py**: Includes the classes necessary to create layers in a multilayer perceptron neural network and to create the neural network containing those layers.
  - **MLPHiddenLayer**: Creates a hidden layer for a multilayer perceptron neural network. A hidden layer has a set size, a weight initialization function, an activation function used by all of the layer's neurons, and an optional optimization function for optimizing weight convergence during backpropagation.
  - **MLPOutputLayer**: Creates an output layer for a multilayer perceptron neural network. An output layer has a set size, a weight initialization function, an activation function used by all of the layer's neurons, a loss function used to calculate the error between the true outputs and the network's predicted outputs, an optional regularization function for keeping gradients from exploding or vanishing, and an optional optimization function for optimizing weight convergence during backpropagation.
  - **MultilayerPerceptronNN**: Creates a full multilayer perceptron neural network with a set input layer size, hidden layer count, hidden layer size(s), and output layer size. The weight initialization, activation, loss, regularization, and optimization functions can be specified for applicable layers. The processing device (cpu/gpu) can also be specified.
- **activation.py**: Includes a class containing neuron activation functions with cpu and gpu support.
  - **Activation**: Class containing the following neuron activation functions and their derivative functions:
    - Leaky ReLU
    - Linear
    - ReLU
    - Sigmoid
    - Softmax
    - Tanh
- **initializer.py**: Includes a class containing weight initialization functions with cpu and gpu support.
  - **Initializer**: Class containing the following weight initialization functions with uniform or normal distributions:
    - Glorot/Xavier
    - He/Kaiming
    - LeCun
- **loss.py**: Includes a class containing loss functions with cpu and gpu support.
  - **Loss**: Class containing the following loss functions:
    - Binary Cross-entropy
    - Categorical Cross-entropy
    - Hinge
    - Huber
    - Kullback-Leibler Divergence
    - Mean Absolute Error (L1)
    - Mean Squared Error (L2)
    - Log-cosh
    - Sparse Categorical Cross-entropy
- **optimizer.py**: Includes a class containing optimization functions with cpu and gpu support.
  - **Optimizer**: Class containing the following optimization functions:
    - None: No optimization performed
    - AdaDelta
    - AdaGrad: Adaptive Gradient
    - Adam: Adaptive Moment Estimation
    - Momentum
    - NAG: Nesterov Accelerated Gradient
    - RMSProp: Root-Mean-Sqaure Propagation
- **regularizer.py**: Includes a class containing regularization functions with cpu and gpu support.
  - **Regularizer**: Class containing the following regularization functions:
    - None: No regularization performed
    - Elastic Net Regerssion
    - LASSO: Least Absolute Shrinkage and Selection Operator (L1)
    - Ridge Regression (L2)
