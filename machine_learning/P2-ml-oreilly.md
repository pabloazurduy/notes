_Some notes from the book [Hands On Machine Learning - OReilly][1]_
_created on: 2022-11-06 11:01:20_
# Hands On Machine Learning Notes - Part 2- DL
## Chapter 10 
### Artificial Neural Networks ANN

The perceptron is one of the simplest ANN architectures, invented in 1957 by Frank Rosenblatt. It is based on a slightly different artificial neuron (see Figure 10-4) called a threshold logic unit (TLU), or sometimes a linear threshold unit (LTU). The inputs and output are numbers (instead of binary on/off values), and each input connection is associated with a weight. The TLU first computes a linear function of its inputs: $z = w_1x_1 + w_2x_2 + ⋯ + w_nx_n + b = w^t x + b$. Then it applies a **step function** to the result: $h_w(x) = step(z)$. 

The most common **step function** used in perceptrons is the Heaviside step function. Sometimes the sign function is used instead.

$$\text{heaviside}(z) = 
\begin{cases}
0  & \text{if} \; z \lt 0 \\
1  & \text{if} \; z \ge 0 
\end{cases}
$$

$$\text{sgn}(z) = 
\begin{cases}
-1  & \text{if} \; z \lt 0 \\
0  & \text{if} \; z = 0 \\
1  & \text{if} \; z \gt 0 
\end{cases}
$$

A perceptron is composed of one or more TLUs organized in a single layer, where every TLU is connected to every input. Such a layer is called a fully connected layer, or a dense layer. The inputs constitute the input layer. And since the layer of TLUs produces the final outputs, it is called the output layer. For example, a perceptron with two inputs and three outputs is represented in Figure 10-5.

<img src="P2-img/perceptron_layer.png" style='height:400px;'>

Computing the outputs of a fully connected layer

$$h_{W,b}(X) =\phi(XW+b)$$

- As always, $X$ represents the matrix of input features. It has one row per instance and one column per feature.
- The weight matrix $W$ contains all the connection weights. It has one row per input and one column per neuron.
- The bias vector $b$ contains all the bias terms: one per neuron.
- The function $\phi$ is called the activation function: when the artificial neurons are TLUs, it is a step function

**Training**

The perceptron is fed one training instance at a time, and for each instance it makes its predictions. For every output neuron that produced a wrong prediction, it reinforces the connection weights from the inputs that would have contributed to the correct prediction. Perceptron learning rule (weight update)

$$w_{i,j}^\text{next step} =w_{i,j}+\eta(y_j−\hat{y_j})x_i$$


- $w_{i, j}$ is the connection weight between the $i^{th}$ input and the $j^{th}$ neuron.
- $x_i$ is the $i^{th}$ input value of the current training instance.
- $\hat{y_j}$ is the output of the $j^{th}$ output neuron for the current training instance.
- $y_j$ is the target output of the $j^{th}$ output neuron for the current training instance.
- $\eta$ is the learning rate (see Chapter 4)

This is very similar to the gradient descent

### The Multilayer Perceptron and Backpropagation

An MLP is composed of one input layer, one or more layers of TLUs called hidden layers, and one final layer of TLUs called the output layer. 

<img src="P2-img/mlp.png" style='height:400px;'>

Is trained using **Backpropagation**. This is how it works:

1. It handles one mini-batch at a time (for example, containing 32 instances each), and it goes through **the full training set multiple times. Each pass is called an epoch** I will re-use the same dataset n-epoch times.
1. Each mini-batch enters the network through the input layer. The algorithm then computes the output of all the neurons in the first hidden layer, for every instance in the mini-batch. The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. This is the _"forward pass"_: it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass.
1. Next, the algorithm measures the network’s output error (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error).
1. Then it computes how much each output bias and each connection to the output layer contributed to the error. This is done analytically by applying the chain rule (perhaps the most fundamental rule in calculus), which makes this step fast and precise.
1. The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the chain rule, working backward until it reaches the input layer. As explained earlier, this reverse pass efficiently measures the error gradient across all the connection weights and biases in the network by propagating the error gradient backward through the network (hence the name of the algorithm).
1. This means that the gradient is calculated once per minibatch, and therefore minibatch/dataset*epoch evaluations. Each epoch contain multiples minibatches
1. Finally, the algorithm performs a gradient descent step to tweak all the connection weights in the network, using the error gradients it just computed.


In order for backprop to work properly, Rumelhart and his colleagues made a key change to the MLP’s architecture: they replaced the step function with the logistic function, $σ(z) = 1 / (1 + exp(–z))$, also called the sigmoid function. This was essential because the step function contains only flat segments, so there is no gradient to work with (gradient descent cannot move on a flat surface), while the sigmoid function has a well-defined nonzero derivative everywhere, allowing gradient descent to make some progress at every step. other activation functions 

1. hyperbolic tangent function $tanh(z) = 2\sigma(2z)-1$
1. ReLU: $ReLU(z)=\max(o,z)$. it works very well and has the advantage of being fast to compute, so it has become the default

![alt text](P2-img/activation_functions.png)



[//]:1.sklearn-oreilly.md> (References)
[1]: <https://github.com/yanshengjia/ml-road/blob/master/resources/Hands%20On%20Machine%20Learning%20with%20Scikit%20Learn%20and%20TensorFlow.pdf>
