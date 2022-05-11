# BPNNP-pytorch
The code is used for constructing interatomic force field within the framework of Behler-Parrinello Neural Network Potential (BPNNP).
If you are enough with [n2p2](https://github.com/CompPhysVienna/n2p2) or RuNNer's slow training, you can take a look.
By using of GPU computing and pytorch's automatic differentiation, the training can be accelerated for at least 20 times.

# Things already done
* Dataloader which accepts ASE database file as the input.
* 4 symmetry functions as defined in [Atom-centered symmetry functions for constructing high-dimensional neural network potentials](https://aip.scitation.org/doi/full/10.1063/1.3553717).
* Fast neighbor list algorithm using GPU.
* Model training with simple full connected neural networks (FCNN).

# Future updates
* ASE machine learning calculator
* Add Bayesian layers in the model
* Batch active learning workflow
