---
layout: post
title: "How to design a machine learning model?"
img: designing-ml-model.png # Add image post (optional)
date: 2017-07-9 12:55:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
tag: [NLP]
comments: true
share: true
---

Today Machine Learning and approaches to conquer the state of AI is The Hottest topic in the world, and actually driving the whole world. We can find 1000 of website talking about how to do machine learning. Basically every-one talks about how to prepare data and input to model and get predictions out. for smaller known data-set this works fine but when applied to larger data-set it may not perform well. To efficiently conquer problem, model designing should be considered as most critical step in the operation. something very similar to Abraham Lincoln's saying "If I had 8 hours to chop down a tree, I would spend 6 of those hours sharpening my axes."

Here we will deep dive in to "How to design a machine learning model?". First we will start with the simplest on XOR gate.

XOR gate is an ideal example because there is no single function that produces a hyper-plane capable of separating the points of the XOR function. The curve in the image separates the points, but it is not a function.

<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_3c625d49039b4f87a78e1d5729aebd09~mv2.png/v1/fill/w_161,h_161,al_c,usm_0.66_1.00_0.01/884a24_3c625d49039b4f87a78e1d5729aebd09~mv2.png">
</p>

<p align="center">Figure 1. Function to solve xor gate.</p>

To separate the points of XOR, you'll have to use at least two lines (or any other shaped functions). This will require two separate perceptrons. Then, you could use a third perceptron to separate the intermediate results on the basis of sign.

<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_8b194d72c0a74e1998cdf0c9d609b6b7~mv2.png/v1/fill/w_161,h_161,al_c,usm_0.66_1.00_0.01/884a24_8b194d72c0a74e1998cdf0c9d609b6b7~mv2.png">
</p>
<p align="center">Figure 2. Two separate linear Function to solve xor gate.</p>
Its very clear that one perceptron that cannot solve XOR function, we will design the smallest network that solves this function.I am a great fan of keras, Lets do some coding then

```python
import keras
import numpy as np
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout,Activation , initializations, regularizers, constraints
x=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]
#multiplying dataset 1000 times
X = []
Y = []
for i in range (0,1000):
    for SignalNo in range(0,len(x)):
        X.append(x[SignalNo])
        Y.append(y[SignalNo])
#converting to numpy array as keras is numpy array hungry
X = np.asarray(X,dtype="int8")
Y = np.asarray(Y,dtype="int8")
# converting to one hot vector
Y = np_utils.to_categorical(Y, 2)
print X.shape, Y.shape
#defining model
model1 = Sequential()
model1.add(Dense(1, input_dim=2, init='uniform', activation='sigmoid'))
model1.add(Dense(2, activation='softmax'))
model1.summary()
model1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
```

The overall network is as given below
<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_a683278170e644b7b7654cc059bf7925~mv2.png/v1/fill/w_227,h_132,al_c,usm_0.66_1.00_0.01/884a24_a683278170e644b7b7654cc059bf7925~mv2.png">
</p>
<p align="center">Figure 3. Simple Network with one hidden perceptron to solve xor gate.</p>

The graph for error and accuracy is as given below, The accuracy will newer improve beyond 75%(It can predict 3/4 cases perfectly but not all)

<p align=   "center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_8ace632932ae46219560d55267329986~mv2.png/v1/fill/w_482,h_262,al_c,lg_1/884a24_8ace632932ae46219560d55267329986~mv2.png">
</p>
<p align="center"> Figure 4. Loss and Accuracy when simple Network with one hidden perceptron used to solve xor gate.</p>

Now lets change this line model

```python
model1.add(Dense(1, input_dim=2, init='uniform', activation='sigmoid'))
```

to

```python
model1.add(Dense(2, input_dim=2, init='uniform', activation='sigmoid'))
```

<p align=   "center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_9cbdadb60dc846e88d16fc162bdd3e70~mv2.png/v1/fill/w_341,h_199,al_c,lg_1/884a24_9cbdadb60dc846e88d16fc162bdd3e70~mv2.png">
</p>
<p align=   "center">Figure 5. Simple Network with two hidden perceptron to solve xor gate.</p>

After running for 100 epochs we will get following graph

<p align=   "center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_ddcad66c4af4464b9c65f17a5b8f0dba~mv2.png/v1/fill/w_473,h_260,al_c,usm_0.66_1.00_0.01/884a24_ddcad66c4af4464b9c65f17a5b8f0dba~mv2.png">
</p>
<p align=   "center">Figure 6. Loss and accuracy when simple Network with two hidden perceptron used to solve xor gate.</p>

At 84 th epoch we got consistent 100% accuracy with lot less error then previous network.
So this topology of the network is said to be perfect.

We will see one more example, which is very complex but it will certainly will provide intuition for modelling a better network not the random one.
I have used a data from news domain where aim is to identify name of the PERSON from text. I have used data from diverse sources one of them was from Cognitive Computing Group. here Clinton and Lazio are name of two persons and so labelled as 1. if such data is provided to deep neural network, Deep Neural Network shold be able to learn adn predict name of person from unknown sources.

```c
Clinton    1
and    0
Lazio    1
would    0
also    0
appeal    0
to    0
independent    0
groups    0
supporting    0
them    0
to    0
refrain    0
from    0
broadcasting    0
ads    0
on    0
their    0
behalf    0
,    0
leaving    0
the    0
campaign    0
committees    0
as    0
the    0
only    0
advertisers    0
in    0
the    0
contest    0
.    0
```

After trying many typologies, I fixed the one which is designed taking human nature in consideration. Looking at data set given above in order to tag one word as person, we require to focus on three aspect of sentence.

  1. The linguistics aspect

     How language is organized, what is the probability of one word being a PERSON, given semantic meaning of the entire sentence. This can be better modeled using a LSTM model.

  2. Features

      Many feature can be given as far as Named entity recognition is concerned. For a particulate word in a sentence. The probability of one word being a PERSON can be decided by following features of the word. for example if word is not the first word in sentence but it in Title case, there is some prbability thet such word can be named entity (Person or Place or Car etc..)
        * is_upper
        * is_lower
        * is_title
        * any(is_upper ) and any(is_lower) # combination of  upper and lower case letter in word
        * is_alphanumeric
        * is_numeric
        * any(is_upper ) and any(is_alphanumeric) # combination of upper and alphanumeric  letter in word
        * any(is_lower ) and any(is_alphanumeric) # combination of lower and alphanumeric letter in word
        * sufiix (if present from list of suffix)
        * prefix (if present from list of prefix)
        * grammer tagging (such as Noun, pronoun, adjective etc.)
        * You can have many more features considering nature of your problem

  3. Word similarity

     Word similarity is a meaningful feature, but we don' t want to use it by implementing distance based algorithms. Best thing would be to convert a word to 2 dimensional sparse matrix and feed to convolution network as it is in native form and let the network to figure out similarity.

# Lets design the network considering 1, 2, 3 points #

1. The linguistics aspect  : sub-network -1

    ```python
    model1 = Sequential()
    model1.add(Convolution1D(64, 3, border_mode='same', input_shape=(12,63)))
    model1.add(MaxPooling1D(pool_length=4))
    model1.add(Convolution1D(nb_filter=32,filter_length=3,border_mode='valid',activation='relu'))
    model1.add(LSTM(256))
    model1.add(Activation('relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(2))
    model1.add(Activation('softmax'))
    model1.summary()
    ```

    Above given sub-network -1 will deal with the linguistic aspect. First layer is of convolution network, which will take 2d matrix as input, will pool out required feature from each maxpooling layer. second layer will again do the same work and provide condensed features to the LSTM (Long Short-Term Memory) layer.

2. Features those we can see:  sub-network - 2

    Typically a word is Person[Entity] or not also depends on a word previous and after to it. Lets say we have identified 12 (as shown above ) features for a particular word. We will take 3 deep belief network each one will take input for the one word for which predictions are to be made a word before and after. Here I have kept input dimension as 12, we have discussed 12 features for a word earlier. those all can be fed here. Its always better to keep track of word before and after the word on which prediction to be made. For the same purpose I have taken three network; 1) one for word before word for which prediction to be made  2) one for word it self for which prediction to be made and 3)  one for word after word for which prediction to be made

    ```python
    model2 = Sequential()# for previous word
    model2.add(Dense(12, input_dim=12, init='uniform', activation='relu'))
    model2.add(Dense(8, init='uniform', activation='relu'))
    model2.add(Dense(1, init='uniform', activation='sigmoid'))

    model3 = Sequential() # for a given word
    model3.add(Dense(12, input_dim=12, init='uniform', activation='relu'))
    model3.add(Dense(8, init='uniform', activation='relu'))
    model3.add(Dense(1, init='uniform', activation='sigmoid'))

    model4 = Sequential() # for the next word
    model4.add(Dense(12, input_dim=12, init='uniform', activation='relu'))
    model4.add(Dense(8, init='uniform', activation='relu'))
    model4.add(Dense(1, init='uniform', activation='sigmoid'))
    ```

    It takes 12 features as input and have 2 hidden layers gives one output.

3. Word Similarity -  sub-network -3

    ```python
    Here also we are going to use 3 network to track word similarity/ patterns similarity for tagging a particular word as Person. To see details of how to represent a word so that network is able to identify similarity/ patterns, goto section 2 of my language representation tutorial.
    model5 = Sequential()# for previous word
    model5.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            input_shape=(1, img_rows, img_cols)))
    model5.add(MaxPooling2D(pool_size=(3, 3)))
    model5.add(Flatten())
    model5.add(Dense(10))
    model5.add(Activation('tanh'))
    # model5.summary()
    model6 = Sequential()# for a given word
    model6.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model6.add(MaxPooling2D(pool_size=(3, 3)))
    model6.add(Flatten())
    model6.add(Dense(10))
    model6.add(Activation('tanh'))
    # model6.summary()
    model7 = Sequential()# for the next word
    model7.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model7.add(MaxPooling2D(pool_size=(3, 3)))
    model7.add(Flatten())
    model7.add(Dense(10))
    model7.add(Activation('tanh'))
    # model7.summary()
    ```

# A larger Network #

A larger network that will keep all sub-networks in sync and flow of error from larger network will flow to all other network through back propagation.
Lets merge all of them

```python
merged = Merge([model1, model2, model3, model4, model5, model6, model7], mode='concat')
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(30))
final_model.add(Activation('tanh'))
final_model.add(Dense(10))
final_model.add(Activation('tanh'))
final_model.add(Dense(2, activation='softmax'))
```

It will add up [concatenate] all the output from smaller sub-networks and then apply smaller fully connected network on the so produced vector.

This network was trained with Adedelta optimizer and categorical cross entropy as loss function.  The F1 score is reported for all networks when trained for 10 epochs and tested on 30% of unexposed test data. It clearly says Ensemble networks are best technique to conquer such problems.

<p align=   "center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_1fbb0ec7f6724cf4aedd839a1d197754~mv2.png/v1/fill/w_870,h_100,al_c,usm_0.66_1.00_0.01/884a24_1fbb0ec7f6724cf4aedd839a1d197754~mv2.png"></p>
<p align=   "center">Figure 7. Confusion Matrix, showing comparison of performance of individual networks and combination of all three (Ensemble)</p>

<p align=   "center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_fdcb4f05472e4a58a8dced68c055d08e~mv2.png/v1/fill/w_643,h_360,al_c,usm_0.66_1.00_0.01/884a24_fdcb4f05472e4a58a8dced68c055d08e~mv2.png"></p>
<p align=   "center">Figure 8. A plot , showing performance of individual networks and combination of all three (Ensemble)</p>

The key take away is to model the network in a way human sense the data not just stacking Perceptrons layer by layer and hoping something to come out as result as miracle.
