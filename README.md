# Mixture Density VAEs

This repository contains implementations of variational autoencoders with Gaussian mixture latent spaces (finite or truncated infinite).  Models and experiments are described in our workshop paper [*Approximate Inference for Deep Latent Gaussian Mixtures*](http://www.ics.uci.edu/~enalisni/BDL_paper20.pdf).

Experiment 1)
train gaussVAE with dataset Fashion MNIST, and compare the likelihood of in-dist(Fashion MNIST test dataset) and OoD (MNIST test dataset)
** python train_reg_gaussVAE.py

Experiment 2)
train gaussVAE with dataset CIFAR10, and compare the likelihood of in-dist(CIFAR) and OoD (SVHN)
** To-Do

Experiment 3)
check if there exists data input complexity bias
