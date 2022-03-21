# digits-nn

This is my implementation of a neural network for identifying
handwritten digits from the MNIST dataset.  
<https://deepai.org/dataset/mnist>

The [nalgebra](https://crates.io/crates/nalgebra) crate is
used for linear algebra.

Some of the code is somewhat spaghetti-ish at the moment.

## Network info

Softmax activation is used for the output layer; ReLU activation
is used for all other layers. The cost function used is categorical
cross-entropy.

Momentum and Nesterov accelerated gradient as described
[here](https://ruder.io/optimizing-gradient-descent/index.html)
are used for learning optimization.

The repository additionally includes a gradient check function
that uses the functionality described
[here](https://cs231n.github.io/neural-networks-3/#gradcheck).

The values I used for the following configurable parameters are:

* hidden layers: 40 and 16 neurons
* learning rate: 0.01
* momentum decay: 0.9
* batch size: 300
* epochs: 100

## Usage

The [RON](https://github.com/ron-rs/ron) format is used for
configuration.

Invoking the binary will attempt to load a file named `network` from
the current directory, and initialize a network with random weights
and zeroed biases if no existing network is found.

Invoking without arguments will train the network based on the config
from a file named `config.ron` in the current directory.  
A thread pool is used to increase training speed.

Invoking with the `--test` argument will run the network on the
entire test dataset, keeping track of average cost and accuracy.

The format of the configuration file is as follows:

* `data`: locations of data
  * `train`: training data
    * `labels`: training labels file
    * `images`: training images file
  * `test`: testing data
    * `labels`: testing labels file
    * `images`: testing images file
* `h_layers`: array representing neuron counts in hidden layers
* `learning_rate`: coefficient of gradient descent steps (`0–1`)
* `momentum_decay`: coefficient of decaying momentum (`0–1`)
* `batch_size`: number of samples for each gradient descent step
* `epochs`: number of times the entire training set is repeated
