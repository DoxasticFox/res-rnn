# ResRNN

A recurrent neural network with residual connections. It sets a new state-of-the-art on pMNIST, achieving 98.45% accuracy, despite its trivially simple design.

## Testing The 4000-dimensional pMNIST Model

```bash
% ./pmnist-test-4000.py
Accuracy of the network on the 10000 test images: 98.45 %
```

## I’m Too Lazy for TikZ but Here’s the Architecture

### The Res-RNN Itself

One thing to note about the residual connection shown below is that [it's weighted](https://github.com/DoxasticFox/res-rnn/blob/c219ebe37e7560a6f512a391c34041af88cfb81f/nnmodules/__init__.py#L76) similarly to in a Gated Recurrent Network (GRU). However, unlike a in GRU, the model isn't allowed to learn what proportion of the output comes from the skip connection and what proportion comes from the previous layer. This shouldn't matter, because the model can learn to make the magnitude of the previous layer large enough to overcome the fixed weight.

![Figure 2](https://github.com/DoxasticFox/res-rnn/raw/master/figures/figure-2.jpg)

### The Res-RNN Applied to a Sequence

Weird stuff is happening here. At each time step, instead of doing additions and/or multiplications to combine the input with the state, like in a regular RNN, I'm "shifting" (like bit shifting) the input into the fixed-size vector which gets fed into the next RNN step. That means that part of the previous state is truncated and ignored. Although I don't see why the model couldn't learn to put all the useful info into the part of the state which is always retained.

![Figure 1](https://github.com/DoxasticFox/res-rnn/raw/master/figures/figure-1.jpg)

## Random

* The 4000-dimensional models gets 98.45% accuracy on pMNIST, even beating fake RNNs, like [TrellisNets](https://arxiv.org/pdf/1810.06682.pdf) and [IGLOO](https://arxiv.org/pdf/1807.03402.pdf).
* The 1000-dimension model gets 98.28% on pMNIST.
* The 200-dimensional model gets 98.78% on standard MNIST. The model is ~326kB on disk, which is 0.25% the size of the 4000-dimensional pMNIST model.
* The learning rate is really, really high (e.g. 1e+5) because of the way the residual connections have been parameterised.
