---
layout: post
title: Introduction To Deep Neural Network
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-05-27
category: articles
tags: [Deep Neural Network]
img: neural_net.png
comments: true
share: true
---

Artificial neural networks were invented in 1943 but could not get popular for application in wider areas. Few reasons Behind failure of ANN are summarized below:

1. Training issues

    Machine needs sufficient representative examples in order to capture the underlying structure that allows it to generalize to new cases. In nineties, Information technology world was flourishing and there was lack of sufficient data for training. After 1970, major pioneer institute start working on the task of gathering data, Image net, word net, dbpedia are result of this. Major companies like Facebook, google, Microsoft started crowd sourcing data from users by providing free image (google image, one-drive), video(YouTube), text(Messenger, Allo, gmail, Hotmail) services. With penetration of mobile in life, data started increasing day by day and we reached at point where data was sufficient. Now with this data algorithm could be trained to achieve certain level of generalization.

2. Theoretical issue

    A neural network can have more than one hidden layers, more hidden layers "builds"new abstractions on top of previous layers. And as we mentioned before, you can often learn better in-practice with larger networks.

    However, increasing the number of hidden layers leads to two known issues:

- Vanishing gradients : As more layers are added back propagation fails to communicate errors from output layer towards input layer. As we have seen in the [Artificial neural network implantation](https://www.machinelearningpython.org/single-post/Neural-Network-Implementation) that we calculate error gradient at each perceptron in each layer, these error gradients become gradually very small as we back propagate to multiple hidden layer. As error gradient vanishes we no more effective gradients, updates to weights cannot be made and in-turn learning cannot takes place with multiple hidden layers.

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_93457120a7514c71a33aea0ce1f3170d~mv2.png/v1/fill/w_360,h_312,al_c,usm_0.66_1.00_0.01/884a24_93457120a7514c71a33aea0ce1f3170d~mv2.png"></p>


    <p align="center">Figure 1. Illustrating vanishing error gradient(δ), while back propagating from output layers to input layers.</p>

- Over-fitting : Smaller data is repeatedly exposed to a network, a machine start over-fitting to it. It means, on the same data machine performed better but perform poor on new data. It can be compared with cramming the task by learning algorithm and do not get generalised well to the newer cases.

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_e7788c23be54473d9dbe04fc02419fda~mv2.png/v1/fill/w_208,h_208,al_c,usm_0.66_1.00_0.01/884a24_e7788c23be54473d9dbe04fc02419fda~mv2.png"></p>


    <p align="center">Figure 2. The green line represents an over-fitted model and the black line represents a regularized model. While the green line best follows the training data, it is too dependent on it and it is likely to have a higher error rate on new unseen data, compared to the black line.(Source - Wikipedia)</p>

3. Hardware issues

    Artificial neural network are costlier in them of computational requirement. I will give you one practical example :

    I am was working with below given Convolution network which is a sub-part of VGG16 network. VGG16 is a huge network with 30+ stacked layers and the full network is having 15 lack+ parameters.

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png/v1/fill/w_473,h_278,al_c,lg_1/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png"></p>


    <p align="center">Figure 3. Architecture of VGG16 Network</p>

    ```python
    model  = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,256,256)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # model.summary()
    ```

    This network with some set of images, it took 240 seconds/epochs on GPU with 3000 core and the same network with same data takes 17100 seconds/Epoch on CPU. For the same algorithm CPU takes 70 times more time compared to GPU.

    GPU are recent development, in-spite of having data and algorithm in place, we cold not apply machine learning to it due to Hardware constrains.

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_5c9234b9140e478da7c51a8d7fcba2d8~mv2.png/v1/fill/w_407,h_499,al_c,usm_0.66_1.00_0.01/884a24_5c9234b9140e478da7c51a8d7fcba2d8~mv2.png"></p>


    <p align="center">Figure 4. Increase in computational power over time.</p>

    Above figure clearly depicts that recently computational power has increased exponentially. Noways hand held devices like Laptops are also equipped with enough power that one can apply machine learning on medium size data-set.
    Insurgence of neural network in twenty first century is due to efficient GP-GPU and GPU computing.

In 2006, a publication by Geoffrey Hinton and Ruslan Salakhutdinov successfully trained a multilayered network with superior performance on MNIST data-set, leads to beginning of deep learning era. Deep learning is simply a modified version of ANN.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_4799f71b3ee346ba8a1d08c2ac4ebfa7~mv2.png/v1/fill/w_303,h_382,al_c,lg_1/884a24_4799f71b3ee346ba8a1d08c2ac4ebfa7~mv2.png"></p>


<p align="center">Figure  5. Showing architecture of ANN to be trained with forward pass and back propagation</p>

As shown in Figure 4. ANN network with few layers and train it with forward pass and back propagation. In deep learning this task is performed in little different way.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_139d42f3cd9546479257de6d75dc983c~mv2.png/v1/fill/w_558,h_787,al_c,usm_0.66_1.00_0.01/884a24_139d42f3cd9546479257de6d75dc983c~mv2.png"></p>


<p align="center">Figure  6. Showing training step in Deep Learning.</p>

As shown in <p align="center">figure, 5.deep learning is having two different steps in learning.</p>

1. Layer-Wise Training - Each hidden layer is separately trained with imaginary output. Each layer is trained until error drops to a constant rate and further training is not possible.

2. Fine-Tuning - Due to pre-training, each layer is having appropriates weights. In this step entire network is trained again to update individual weight in a manner that final error of the network is much lesser then individual layer's error and predictions are also improved compared to individual layer's predictions.

3. Advance optimization techniques 

    Now we have a good algorithm in hand, but it takes great amount of time to train a Deep Neural Network if it is not optimized well. It is also possible that the algorithm may not converge/generalized at all without optimization applied. Advance  optimization techniques, played major role in decreasing training time and making network more generalized.

- Dropouts

    Dropout is simple way to prevent algorithm from over fitting and co-adaptation. In drop out , some connection between two layers are randomly dropped at each iteration. This would create a variable network graph and prevent network from over-fitting. Variable graph means all connection has equal opportunity to grow and eventually generalize the network well

    I have given an illustration of drop-out in below given Flow chart, where I have cancelled certain connection between two perceptron of two layers. Such dropping of weight occur at every iteration. Generally 50% of input weight are dropped and 20% of hidden weights are dropped randomly.

    Dropouts algorithm can be implemented in two ways:

  - Memory Friendly Approach - Change matrix size at each iteration as per the number of connected perception between two layers

    Pros : As nearly 20% of connection will be dropped, matrix size will also be decreased accordingly and will become easy to deal with in limited memory. 

    Cons : 1) Frequent matrix initialization would be required, in-turn degradation in speed. 2)  Keeping track of alive and dropped connections is difficult while making update in next iteration. 3) In Matrix it is difficult to keep track that which weight is representing connection between which two perceptions of two layer, To track this a reference matrix is required which of size of totally connected network and resides in hard drive.

  - Speed Friendly Approach -  just multiply dropped connection with zero in matrix

    Pros - 1) No alteration in matrix size. 2) favorable for GPU computations 

    Cons - Memory utilization is same as of fully connected network

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_7bc49135134e462c87a8e7766e250a0a~mv2.png/v1/fill/w_577,h_814,al_c,usm_0.66_1.00_0.01/884a24_7bc49135134e462c87a8e7766e250a0a~mv2.png"></p>


<p align="center">Figure  7. Showing training step to be followed for Deep Learning with Dropout. (Some connections are dropped) </p>

- Adadelta

    Adadelta means adaptive learning rate. This topic I have explained with working example in my earlier blog post.
    Generally we keep learning rate constant throughout the training leads to longer training time. Alternatively we can keep variable learning rate that increases when there is steep decrease in error and learning rate decreases when there id less increase in error. Below given diagram from my earlier post clearly depicts change in learning rate as training progresses, learning rate was high initially but when it reached towards convergence the learning rate decrease and become minimal.

- Nesterov Accelerate Gradient

    Nesterov’s Accelerated Gradient descent (Nesterov Y, 2007), which accelerated the optimization process to reach global minima using less computational complexity over time. The plain Gradient Descent algorithm has a rate of convergence as $D x  y$, while the Nesterov's Accelerated gradient has rate of convergence as $O(1/(t)^2)$

    With this things in mind, I sum up this post, in next post will see about how to apply deep neural network practically to real life problems.
