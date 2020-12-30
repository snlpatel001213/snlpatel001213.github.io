---
layout: post
title: String Matching using Convolution Networks
date: 2017-07-9 12:54:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
tag: [keras, GPU, Python, Machine Learning Advance]
img: stringMatching.png
comments: true
share: true
---

----

Codes discussed in present tutorial can be found on this [GitHub](https://github.com/snlpatel001213/algorithmia/tree/master/stringMatching) repository.
Code Compatibility : Python 2.7, tested on ubuntu 16.04 with theano as backend

----

String matching is a very conventional problem in computational science. String matching is actually a pattern matching, when a portion p need to be found out from another larger string where p and can be of same length and p may not be an exact sub-string. I am not going much deeper in to history and classification of string matching algorithms. Various string matching algorithms can be used for specific purpose. Specialized algorithms are available for partial, multiple and exact string matching. All algorithms works for specific purpose, but single algorithm doesn't do well on all type of string matching. e.g. Levenshtein distance algorithm perform well on string with insertion and deletion but doesn't do well on string with translocation of fragments.
The purpose behind this tutorial is to train machine for matching string having any type of differences (e.g. insertion, deletion and fragment translocation).
I have not used any specialized data-set for this tutorial, in fact synthetic data-set was generated for training.

1. **Data-set Generation :** 
    Data-set is constituted by Original string and a Mutated string:
    Original string is made up of random 100 characters e.g.

    *TSKCG498ZTQR7F5VFI59CSVKFY3XG98OG762HYDF82XB1LM87WNNM5Z57L9DIHFI64W4MSYJ2KK3B17HMJBQZJNVKDVAL0I42ZOX*

    Mutated string is made from  Original string with some noise introduced

    *TJKCG4BqQTZzmF5VlI59CSnKZE3Tk98OG76iHYDF8ZXBcbM8cWNNM5s5FQ9DWHFI6PWnPSEJpmwlB1qAMJIZZGxjKDmAJGI42ZkS*

    In above example  Original string and a Mutated string are 50% similar, it means out of 100 character in original string 50 are changed randomly (point mutation)
    for to constitute data-set, alphanumeric original and mutated strings are randomly generated with random 1 to 100 % mutation.
    For to generate data-set.  I have taken only capital letters (26) + digits (10). The combination for Number of sample points in set = 100  and Number of sample points in each combination = 36 turn out to be 1.97720458214493E+27. The combination so achieved is an astronomic number and probability of getting a string repeated is absolutely null. 

    ```python
    stringSize = 100  # defining string size
    maxWordLength = 100 
    # lets consider 63 basic character, those occur the most
    # character to integer mapping dictionary
    charToInt = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'a': 11, 'B': 12,
                'b': 13, 'C': 14, 'c': 15, 'D': 16, 'd': 17, 'E': 18, 'e': 19, 'F': 20, 'f': 21, 'G': 22, 'g': 23, 'H': 24,
                'h': 25, 'I': 26, 'i': 27, 'J': 28, 'j': 29, 'K': 30, 'k': 31, 'L': 32, 'l': 33, 'M': 34, 'm': 35, 'N': 36,
                'n': 37, 'O': 38, 'o': 39, 'P': 40, 'p': 41, 'Q': 42, 'q': 43, 'R': 44, 'r': 45, 'S': 46, 's': 47, 'T': 48,
                't': 49, 'U': 50, 'u': 51, 'V': 52, 'v': 53, 'W': 54, 'w': 55, 'X': 56, 'x': 57, 'Y': 58, 'y': 59, 'Z': 60,
                'z': 61, ' ': 62, '.': 63}
    # integer to character mapping dictionary 
    intToChar = {v: k for k, v in charToInt.iteritems()}

    def string_generator(size=stringSize, chars=string.ascii_uppercase + string.digits):
        """
        will generate random string
        """
        return ''.join(random.choice(chars) for _ in range(size))
    def mutator(originalString, percentageMutation):
        """
        will take a string and mutate it as per percentage specified
        """
        originalStringArray = list(originalString)
        for i in range(percentageMutation):
            # print originalStringArray
            randomPlace = random.randint(0, len(originalString) - 1)
            randomLetter = random.choice(string.letters)
            originalStringArray[randomPlace] = randomLetter
        return "".join(originalStringArray)


    ```


2. **String Representation :**
    In present tutorial we are going to utilize convolution network. Convolution network are found to produce state of are result in area of image processing. As per my intuition, if strings are represented in form of image then convolution network should performs better there too. 
    Similar images often having point or local differences, still Convolution network perform better. Similarly string  may have point local differences. Convolution network also perform better on images with horizontal/vertical shifts and rotational difference. These difference are similar to string with transnational differences.

    ```python
    def giveWordmatrix(word):
        """
        will generate 2d matrix of the string, which will be an input to convolutional network
        word : is a string given to function
        """
        # 2d matrix of size 100*63 initilaized with all cell having value "false"
        tempMatrix = np.zeros((maxWordLength, 63), dtype=bool)
        charNo = 0


        for charNo in range(0, len(word)):
            if charNo < maxWordLength:
                try:
                    try:
                        # for above defined 63 character, if character exists then "true" is placed in place 
                        characterToIndex = int(word[charNo])
                        tempMatrix[charNo][characterToIndex] = True
                        charNo += 1
                    except:
                        characterToIndex = charToInt[word[charNo]]
                        tempMatrix[charNo][characterToIndex] = True
                        charNo += 1
                except:
                    tempMatrix[charNo][0] = False
        return tempMatrix

    # lets do little visualization
    # generating new string
    originalString = string_generator()
    # print originalString
    # mutating the same string randomly
    prcentageMutation = random.randint(0, 100)
    mutatedString = mutator(originalString, prcentageMutation)
    # genearting 2d matrix for original string
    originalStringMatrix = giveWordmatrix(originalString)
    # genearting 2d matrix for mutated string
    mutatedStringMatrix = giveWordmatrix(mutatedString)
    # visualizing original and muataed string
    print ("Original String")
    pyplot.imshow(toimage(originalStringMatrix))  # showing first image
    pyplot.show()
    print ("Mutated String")
    pyplot.imshow(toimage(mutatedStringMatrix))  # showing first image
    pyplot.show()
    ```
    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_9710badc39f845ccaeb32da04803727e~mv2.png/v1/fill/w_532,h_378,al_c,lg_1/884a24_9710badc39f845ccaeb32da04803727e~mv2.png"></p>
    <p align="center">Figure 1. Original and mutated string is shown as in form of image</p>
    Above image was generated with about 50% mutation, you can clearly see, about 50% points are same and remaining are different. For each such 2D representation, on vertical axis 0 - 100 represent each character in string, each character is mapped to corresponding number between 0 – 63 as defined in dictionary above.

3. **Model Architecture :**
    I always believe in designing model, in the similar way as human thinks. Consider an scenario when we have been given two strings as above and asked to find similarity between them. How we think?  Repetitively see two strings and try to find out difference between them. Right??. 
    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_a532eda2af6244069ac128e419300632~mv2.png/v1/fill/w_945,h_434,al_c,lg_1/884a24_a532eda2af6244069ac128e419300632~mv2.png"></p>
    <p align="center">Figure 2. Model Architecture for string matching task</p>
    Our model architecture is having two model works in parallel. Each model using convolution network generate a condensed representation of the given strings and finally such representation are merged to final model which is stacked dense layer of deep neural network. The final out put is class value between 0 to 100, class value also represents similarity between two strings. By the way this network architecture is known as [Siamese Network](https://www.coursera.org/lecture/convolutional-neural-networks/siamese-network-bjhmj).

    While training such model, two strings are provided, one to each model. Lets say string 1 was provided to model A and mutated string 2 with 20% mutation is provided to model B. The reference class value provide in such case would be 80 (100 – 20), represents similarity between two strings.
    The overall model definition in keras is as given below :

    ```python
    # defining model
    # model2 will take original string 
    model2 = Sequential() 
    model2.add(Convolution1D(64, 3, border\_mode='same', input\_shape=(100, 63,))) 
    model2.add(MaxPooling1D(pool_length=2)) 
    model2.add(Convolution1D(nb\_filter=128,filter\_length=3,border_mode='valid',activation='relu')) 
    model2.add(Flatten()) 
    model2.add(Dense(64, activation='relu')) 
    model2.add(Dropout(0.5)) 
    model2.add(Dense(32, activation='sigmoid')) 
    # model2.summary()
    # model1 will take mutated string 
    model1 = Sequential() 
    model1.add(Convolution1D(64, 3, border\_mode='same', input\_shape=(100, 63,))) 
    model1.add(MaxPooling1D(pool_length=2)) 
    model1.add(Convolution1D(nb\_filter=128,filter\_length=3,border_mode='valid',activation='relu')) 
    model1.add(Flatten()) 
    model1.add(Dense(64, activation='relu')) 
    model1.add(Dropout(0.5)) 
    model1.add(Dense(32, activation='sigmoid'))
    # both model merges  
    merged = Merge([model1, model2], mode='concat') 
    # final model will decide final result 
    final_model = Sequential() 
    final_model.add(merged) 
    final_model.add(Dense(64)) 
    final_model.add(Activation('tanh')) 
    final_model.add(Dense(64)) 
    final_model.add(Activation('tanh'))
    final_model.add(Dense(100, activation='sigmoid')) 
    final_model.summary()
    # compiling model 
    final_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['mse'])
    ```
    
    we will be using stochastic gradient descent as optimizer.  Categorical cross-entropy was used as loss  function as it is performs better in multi class classification. Instead of accuracy I have used mean squared error  as performance metrics. There is a specific reason behind this trick, as this is multi-class class classification you would not get higher accuracy and low accuracy discourages, but one can see clear decrease in mean squared error  after each iteration. In this problem our aim is to predict exact similarity between two strings, if not possible then predicted similarity should be closest to original class.
4. **Actual Training**
    Unlike other tutorial we do not have exact data-set on which we iterate repeatedly and try to fit.
    New data-set is generated at each iteration by repeatedly calling   string_generator and random mutations are performed on it. In this manner no data is repeated for training. This practice makes model versatile, however takes longer training time.
    Please go though below given code, it is to the point and easily understandable

    ```python  # file to watch intermediate results 
    testFileOut = open("intermediate_results.txt", "w") 
    # repeate for 5000 iterations, you may change this 
    for times in range(10000):
        originalStringArray  = []  # to keep original strings 
        mutatedStringArray = []  # to keep mutated strings 
        percentageSameArray = []
        response = []  # to keep percentage simillarity between original and mutated strings 
        print times
        # every time new 10000 strings and their mutated strings are generated and kept in RAM 
        for batchOf in range(10000):
            # generating origianl string 
            originalString = string_generator()
            # randomly deciding percentage mutation 
            prcentageMutation = random.randint(1, 100)
            # 100(stringSize) - mutation = percentage simillarity between original and muatated string 
            percentageSame = 100 - prcentageMutation
            percentageSameArray.append(percentageSame)
            # generating mutated string 
            mutatedString = mutator(originalString, prcentageMutation)
            # generating original string matrix 
            originalStringMatrix = giveWordmatrix(originalString)
            # appending original string matrix to originalStringArray 
            # after 10000 loops, originalStringArray will be having marix for 10000 original strings 
            originalStringArray.append(originalStringMatrix)
            # generating mutated string matrix 
            mutatedStringMatrix = giveWordmatrix(mutatedString)
            # appending mutated string matrix to mutatedStringArray 
            # after 10000 loops, mutatedStringArray will be having marix for 10000 muated strings 
            mutatedStringArray.append(mutatedStringMatrix)
            # response vector is having %simillarity  between original and muatated string 
            # after 10000 loops, response will be having % for above geenrated 10000 original and  
            # corrosponding mutated strings 
            response.append(percentageSame)
        if times % 1000 == 0:
            # at every 1000 iteration it will dump output to testFileOut; this is to see progress of learning 
            # converting originalStringArray having 10000 strings to numpy boolean array 
            originalStringArray =  np.asarray(originalStringArray, dtype='bool')
            # changed nothing in reshape; precautionary  
            originalStringArray = originalStringArray.reshape(originalStringArray.shape[0], originalStringArray.shape[1],
                                                              originalStringArray.shape[2])
            # converting mutatedStringArray having 10000 mutated strings to numpy boolean array 
            mutatedStringArray =  np.asarray(mutatedStringArray, dtype='bool')
            # changed nothing in reshape; precautionary  
            mutatedStringArray = mutatedStringArray.reshape(mutatedStringArray.shape[0], mutatedStringArray.shape[1],
                                                            mutatedStringArray.shape[2])
            # converting respose vector to categorical "one hot encoding"  
            # when we use categorical_crossentropy as loss function, converting to "one hot encoding" is must. 
            response = np_utils.to_categorical(response, 100)
            # training 
            final_model.fit([originalStringArray, mutatedStringArray], response,
                            batch_size = 10000, nb_epoch = 1, verbose = 2, validation_split = 0.2)
            # getting probability for intermediate inspection 
            prob =  final_model.predict_classes([originalStringArray, mutatedStringArray], verbose=0)
            # writting to file 
            for eachNo in range(0, len(list(prob))):
                testFileOut.write(str(prob[eachNo]) + "t" + str(percentageSameArray[eachNo]) + "n")
            testFileOut.flush()
        else:
            # When in pection is not required 
            # converting originalStringArray having 10000 strings to numpy boolean array 
            originalStringArray =  np.asarray(originalStringArray, dtype='bool')
            # changed nothing in reshape; precautionary  
            originalStringArray = originalStringArray.reshape(originalStringArray.shape[0], originalStringArray.shape[1],
                                                              originalStringArray.shape[2])
            # converting mutatedStringArray having 10000 mutated strings to numpy boolean array 
            mutatedStringArray =  np.asarray(mutatedStringArray, dtype='bool')
            # changed nothing in reshape; precautionary  
            mutatedStringArray = mutatedStringArray.reshape(mutatedStringArray.shape[0], mutatedStringArray.shape[1],
                                                            mutatedStringArray.shape[2])
            # converting respose vector to categorical "one hot encoding"  
            # when we use categorical_crossentropy as loss function, converting to "one hot encoding" is must. 
            response = np_utils.to_categorical(response, 100)
            # training 
            final_model.fit([originalStringArray, mutatedStringArray], response,
                            batch_size = 10000, nb_epoch = 1, verbose = 1, validation_split = 0.2)
    ```

    When I stated training, initially model was clueless about data and ended up predicting one similarity value for all combination of original and mutated strings.

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_227dcfb4d0ca4881b34f29a8e4f13cf8~mv2.jpg/v1/fill/w_726,h_408,al_c,lg_1,q_80/884a24_227dcfb4d0ca4881b34f29a8e4f13cf8~mv2.webp"></p>
    <p align="center">Figure 3. Represents predicted and actual difference in strings with un-trained model</p>
    After training model was well learned about the task given and predicted very well on  randomly generated original and mutated strings.
    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_900505a09bce476ab637f61303ba7e24~mv2.jpg/v1/fill/w_821,h_520,al_c,lg_1,q_85/884a24_900505a09bce476ab637f61303ba7e24~mv2.webp"></p>
     <p align="center">Figure 4. Represents predicted and actual difference in strings with trained model</p>
5. **Drill Down**
    I have conducted an experiment to compare performance of our neural network based string match and Levenshtein distance algorithm. A set of 100 original and mutated string was generated randomly and given to  neural network based string match and Levenshtein distance algorithm. I have sorted the result obtained in ascending order of similarity. It is great to note that Levenshtein algorithm fails with string having higher dissimilarity (>75% dissimilarity). However  neural network based string match performed equally well for strings with extreme similarity and dissimilarity.

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_74770b21e7754b2caf81a2f6f0ab9f67~mv2.png/v1/fill/w_917,h_485,al_c,usm_0.66_1.00_0.01/884a24_74770b21e7754b2caf81a2f6f0ab9f67~mv2.png"></p>
    <p align="center">Figure 5. A comparison of predicted string  match with predicted match by Levenshtein Algorithm [two string with insertion deletion and updates] </p>
    In above given tutorial I have used only point mutation. To see the adaptability of the algorithm I have conducted another set of experiment. Along with point mutation, in the next part of tutorial I have applied random translocation in mutated string.
    Below is the example of two string having nearly 80% similarity  with point mutation [In Maroon]  and translocation difference [Red and Green].

    Original String :
    TQITTPC8RVRX6PUSP0782COJV3IHTBDS9KNFSAPS8RI30F5BEYGFCLRH06UF2KTK1EUM776OIITVWM2MDJCKQ6GBACSHTSZ85XOP

    Muated String :
    KTKrVjMU76yIITwWM2SAPS8RI30F5BEYGFCLRH06UF28wCOJV3IHTBDS9KNFTQITTPC8RVRX6PUSP07MDJCKQ6kBAuSHTSZ85XOP

    To generate such strings I have modified mutator function a little and have added  makeRandomFragments which will make random fragment  of given string and randomly concatenate these fragments to form a new string.

    ```python
    def makeRandomFragments(string):
        """ 
        will make random fragment  of given string and randomly concatenate these fragments to form a new string. 
        :param string:  
        :return:  
        """
        splitted = []
        prev = 0
        while True:
            n = random.randint(10, 25)
            splitted.append(string[prev:prev + n])
            prev = prev + n
            if prev >= len(string) - 1:
                break
        return "".join(list(set(splitted)))


    def mutator(originalString, percentageMutation):
        """ 
        will take a string and mutate it as per percentage specified 
        """
        originalStringArray = list(originalString)
        for i in range(percentageMutation):
            # print originalStringArray 
            randomPlace = random.randint(0, len(originalString) - 1)
            randomLetter = random.choice(string.letters)
            originalStringArray[randomPlace] = randomLetter
        return makeRandomFragments("".join(originalStringArray))

    ```

    To train a new model to adapt to learn translocation related changes in the string, I have trained a new model by a method called as transfer learning. In transfer learning some previous model which was trained for the similar task to the current one is taken and weight are readjusted  by retraining . This method takes less time in comparison to make model learn from beginning.
    After completing the training I have compared result of neural network based method  with Levenshtein algorithm. What i found was amazing ,  Levenshtein algorithm performed very poorly on randomly string having high simillarity but randomly trans-located and on string having high mutation. But our method worked well on both parts. 
    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_7b9c1ddb0a5e49e48a796925565b988a~mv2.png/v1/fill/w_945,h_493,al_c,usm_0.66_1.00_0.01/884a24_7b9c1ddb0a5e49e48a796925565b988a~mv2.png"></p>
    
    <p align="center">Figure 6. A comparison of predicted string  match with predicted match by Levenshtein Algorithm [two string with translation if fragments along with insertion deletion and updates]</p>
    You can clearly see yellow line [Levenshtein method](https://en.wikipedia.org/wiki/Levenshtein_distance)  fails on strings having 1) High similarity but random translocation and; 2) Strings having high mutation. In both the cases our method was better adapted.