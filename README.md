# ResRNN

A recurrent neural network with residual connections.

## Random

* The 4000-dimensional models gets SOTA (98.45%) on pMNIST, even beating fake RNNs, like [TrellisNets](https://arxiv.org/pdf/1810.06682.pdf) and [IGLOO](https://arxiv.org/pdf/1807.03402.pdf).
* The 1000-dimension model gets 98.28% on pMNIST.
* The 200-dimensional model get 98.78% on MNIST.
* The learning rate is really, really high (e.g. 1e+5) because of the way the residual connections have been parameterised.
