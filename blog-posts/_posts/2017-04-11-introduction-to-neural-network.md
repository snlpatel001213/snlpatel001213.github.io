---
layout: post
title: iIntroduction to nNeural network
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-07-17
category: articles
tags: [Artificial Neural Network]
img: neural_network.png
comments: true
share: true
---

---

This blog is theoretical and little mathematical Explanation of working of Artificial Neural networks. Try to understand as much as you can, In next tutorial I will walk you through step by step implementation of neural network.

---

A neural network can be defined as a model of reasoning based on the human brain.  The brain consists of a densely interconnected set of nerve cells, or basic information-processing units, called neurons.
The human brain incorporates nearly 10 billion neurons and 60 trillion connections, synapses, between them.  By using multiple neurons simultaneously, the brain can perform its functions much faster than the fastest computers in existence today.
Each neuron has a very simple structure, but an army of such elements constitutes a tremendous processing power.
A neuron consists of a cell body, soma, a number of fibers called dendrites, and a single long fiber called the axon.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_78f2d1230d0345fe94e245974e2f6157~mv2.jpg/v1/fill/w_482,h_203,al_c,q_80/884a24_78f2d1230d0345fe94e245974e2f6157~mv2.webp"></p>

<p align="center">Figure 1. Biological neural network</p>

Our brain can be considered as a highly complex, non-linear and parallel information-processing system.
Information is stored and processed in a neural network simultaneously throughout the whole network, rather than at specific locations.  In other words, in neural networks, both data and its processing are global rather than local.
Learning is a fundamental and essential characteristic of biological neural networks.  The ease with which they can learn led to attempts to emulate a biological neural network in a computer.

An artificial neural network consists of a number of very simple processors, also called neurons, which are analogous to the biological neurons in the brain.
The neurons are connected by weighted links passing signals from one neuron to another.
The output signal is transmitted through the neuron’s outgoing connection.  The outgoing connection splits into a number of branches that transmit the same signal.  The outgoing branches terminate at the incoming connections of other neurons in the network.
Architecture of the Artificial Neural Network is very similar to the Neural Networks found in our brain. You can see one neuron can have my input and one output. all connection have weights associated with it. In figure 3. w1, w2, w3 are input weights to perceptron.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_179b516d59d0495c88f67bb1e88ce829~mv2.jpg/v1/fill/w_445,h_253,al_c,lg_1,q_80/884a24_179b516d59d0495c88f67bb1e88ce829~mv2.webp"></p>

 <p align="center">Figure 2. Architecture of a typical artificial neural network</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_eca8e89787dd4818a8bba688f774e3d5~mv2.jpg/v1/fill/w_445,h_201,al_c,q_80,usm_0.66_1.00_0.01/884a24_eca8e89787dd4818a8bba688f774e3d5~mv2.webp"></p>

 <p align="center">Figure 3. The neuron as a simple computing element</p>

The neuron computes the weighted sum of the input signals and compares the result with a threshold value, Θ.  If the net input is less than the threshold, the neuron output is –1.  But if the net input is greater than or equal to the threshold, the neuron becomes activated and its output attains a value +1. Here Y one of the class to be predicted and X is the actual output from output neuron.This type of activation function is called a sign function.

For example your neural output is 0.56 and you threshold is 0.5 then as 0.56 > 0.5, class is said to be 1.

The neuron uses the following transfer or activation function:
<center>

$$X = \sum_{i=1}^n xiwi
\quad \quad \quad Y  = \left\{\begin{aligned}
+1 \quad if \quad  X >= \theta\\-1 \quad if \quad X < \theta
\end{aligned}
\right.$$
</center>

<p align="center">General Meaning of activation function</p>

Beside sign activation function there are other activation function exist as given below:

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_b2d44f33d5324bdf8d2467c7d5ad4e03~mv2.png/v1/fill/w_700,h_378,al_c,lg_1/884a24_b2d44f33d5324bdf8d2467c7d5ad4e03~mv2.png"></p>

<p align="center">Figure. 4 Activation functions of a neuron</p>

In 1958, Frank Rosenblatt  introduced a training algorithm that provided the first procedure for training a simple ANN: a perceptron.  The perceptron is the simplest form of a neural network.  It consists of a single neuron with adjustable synaptic weights and a hard limiter (activation used).

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_a92492af71784fdda7adf7d72fcd5a6a~mv2.png/v1/fill/w_700,h_371,al_c,lg_1/884a24_a92492af71784fdda7adf7d72fcd5a6a~mv2.png"></p>

<p align="center">Figure. 5 Single-layer two-input perceptron</p>

The operation of Rosenblatt’s perceptron is based on the McCulloch and Pitts neuron model.  The model consists of a linear combiner followed by a hard limiter (a type of activation function).
The weighted sum of the inputs is applied to the hard limiter, which produces an output equal to +1 if its input is positive and 1 if it is negative.
The aim of the perceptron is to classify inputs, $ x1, x2, . . ., xn,$ into one of two classes, say  A1 and A2.
In the case of an elementary perceptron, the n-dimensional space is divided by a hyperplane into two decision regions.  The hyperplane is defined by the linearly separable function:

To understand below given figure I will walk you through one example: lets say we have to predict house price that depend on two variables x1 and x2 then we have two dimensional plane (Figure 7A )of results (just like we plot 2D  graph with ). When we add one more factor x3 then search dimensional becomes 3 dimensional (Figure 7B). A swe go on adding more and more variables (features), we will be able to conquer more complex spaces and such solutions can handle non linearity well.
We will also see one example as what is the impact of having smaller network trying to predict bigger problem.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_107ec9e048b14750b2531c6397b3cb6a~mv2.png/v1/fill/w_700,h_397,al_c/884a24_107ec9e048b14750b2531c6397b3cb6a~mv2.png"></p>

<p align="center">Figure 6. Linear separability in the perceptrons</p>

**How does the perceptron learn its classification tasks?**
This is done by making small adjustments in the weights to reduce the difference between the actual and desired outputs of the perceptron.  The initial weights are randomly assigned, usually in the range [-0.5, 0.5], and then updated to obtain the output consistent with the training examples.

**What is actually back propagation:**
The network computes its output pattern, and if there is an error  or in other words a difference between actual and desired output patterns  the weights are adjusted to reduce this error.
In a back-propagation neural network, the learning algorithm has two phases.
First, a training input pattern is presented to the network input layer.  The network propagates the input pattern from layer to layer until the output pattern is generated by the output layer.
Second, If this pattern is different from the desired output, an error is calculated and then propagated backwards through the network from the output layer to the input layer.  The weights are modified as the error is propagated.


If at iteration p, the actual output is Y(p) and the desired output is Yd (p), then the error is given by:

<center>

$ e(p) = Y_d(p) - Y(p)$
</center>
<p align="center">Error calculation</p>

where p = 1, 2, 3, . . .
Iteration p here refers to the pth training example presented to the perceptron. If the error, $e(p)$ , is positive, we need to increase perceptron output $Y(p)$, but if it is negative, we need to decrease $Y(p)$ .

<center>

$\Delta W_i(p) = \alpha*X_i(p)*e(p)$
</center>

<p align="center">Updates to weights to decrease error Δw</p>

where $p$ = 1, 2, 3, . . .
α is the learning rate, a positive constant less than unity.The perceptron learning rule was first proposed by Rosenblatt in 1960. Using this rule we can derive  the perceptron training algorithm for classification  tasks.

I recommend you to read this paper if you are looking for more mathematically and theoretical formalized version.
This is the shortest theory about working of neural networks. in upcoming tutorial we will see how to practically implement the Artificial Neural Network Algorithm.
