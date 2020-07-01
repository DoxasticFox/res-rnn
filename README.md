# ResRNN

A recurrent neural network with residual connections. It gets SOTA on pMNIST.

## Random

* The 4000-dimensional models gets 98.45% accuracy on pMNIST, even beating fake RNNs, like [TrellisNets](https://arxiv.org/pdf/1810.06682.pdf) and [IGLOO](https://arxiv.org/pdf/1807.03402.pdf).
* The 1000-dimension model gets 98.28% on pMNIST.
* The 200-dimensional model gets 98.78% on standard MNIST. The model is ~326kB on disk, which is 0.25% the size of the 4000-dimensional pMNIST model.
* The learning rate is really, really high (e.g. 1e+5) because of the way the residual connections have been parameterised.

## Testing The 4000-dimensional pMNIST Model

```bash
% ./pmnist-test-4000.py
Accuracy of the network on the 10000 test images: 98.45 %
```

## I’m Too Lazy for TikZ but Here’s the Architecture

### The Res-RNN Applied to a Sequence

![Figure 1](https://github.com/DoxasticFox/res-rnn/raw/master/figures/figure-1.jpg)

### The Res-RNN Itself

![Figure 2](https://github.com/DoxasticFox/res-rnn/raw/master/figures/figure-2.jpg)
