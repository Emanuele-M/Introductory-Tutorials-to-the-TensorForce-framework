# TensorForce Tutorial Development

This repository contains some tutorials covering the implementation of Reinforcement Learning agents using the TensorForce framework, as well as an introductory tutorial to some of the perks and features of this RL framework.

# Getting started

In order to use these tutorials, OpenAI Gym and the TensorForce framework need to be installed.

To install OpenAI Gym, please follow the instructions on [Gym's github repository](https://github.com/openai/gym).

To install TensorForce, follow the instructions on the [TensorForce official website](https://tensorforce.readthedocs.io/en/latest/basics/installation.html).

# Tutorials

In this repository are collected three tutorials whose aim is to introduce the TensorForce framework, its features and the main abstractions provided for various parts of the main elements of Reinforcement Learning, such as agents, environments and modules, as well as its introduction of an execution routine, and showcase these qualities through two practical applications over environments from the Open AI Gym toolkit.

* Introduction to the TensorForce framework
 > This tutorial is meant to serve as an introduction to the various features provided by the TensorForce framework. In it are discussed the essential functions and features characterizing each different element for which an abstraction is provided (agents, environments, modules and execution) and is then reported a basic example in which an extremely simple agent is created, trained and evaluated in a basic environment.

* [Vanilla Policy Gradient agent in CartPole environment](https://github.com/Emanuele-M/Progetto-Tesi-Tensorforce/blob/main/Vanilla%20Policy%20Gradient%20-%20CartPole/Policy%20Gradient%20agent%20implementation%20in%20TensorForce.ipynb)
 > This tutorial consists of a detailed explaination of the TensorForce implementation of a Reinforcement Learning agent based on the Vanilla Policy Gradient (or REINFORCE) algorithm and its usage to solve the CartPole environment from OpenAI Gym. It also reports a sufficiently detailed introduction to the concept of *policy* and to Policy Gradient methods.
 > <div><img src=\"78819170-cb8f0780-79a3-11ea-8ad6-069968da4d14.gif\", width=500px, height=500px></div>

* [Policy Gradient solution to the LunarLander environment using the TensorForce framework](https://github.com/Emanuele-M/Introductory-Tutorials-to-the-TensorForce-framework/blob/main/Policy%20Gradient%20-%20LunarLander/Policy%20Gradient%20solution%20to%20the%20LunarLander%20environment%20using%20the%20TensorForce%20framework.ipynb)
 > This tutorial reports a detailed walkthrough of the creation, training and evaluation of an agent in the LunarLander environment from the Open AI Gym toolkit, with particular attention given to explaining the significance of the values and parameter configurations used in the example. It also contains a brief summary of the idea behind Policy Gradient methods and, in particulare, describes the REINFORCE algorithm.
 > <div><img src=\"3.gif\", width=500px, height=500px></div>
