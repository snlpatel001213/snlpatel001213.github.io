---
layout: post
title: SMOTE - Synthetic Minority Over-sampling Technique
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-05-27
category: articles
tags: [Machine Learning Basic, Python]
img: smote.png
comments: true
share: true
---

----

Codes described throughout are placed in [this](https://www.machinelearningpython.org/single-post/2017/04/25/t-Distributed-Stochastic-Neighbor-Embedding-t-SNE-in-python) GitHub Repository.

For theoretical information and pseudo code you may refer : [https://www.jair.org/media/953/live-953-2037-jair.pdf](https://www.jair.org/media/953/live-953-2037-jair.pdf)

---

Imbalance in the data-set is mitigated using SMOTE. suppose we have a data-set one class is nearly 85% and other class is nearly 15%. This imbalance is not good for machine learning. Machine learning generally prefers same number of sample for all classes for effective balanced learning and prediction. I have worked with data-set having one class up-to 97% and other is about 3% only. I know the headache if dealing with such data-set. Certain techniques can be used to balance such biased data-set by synthesizing new samples identical to minor class.
For this tutorial we will be using iris data-set, Iris data-set is having 3 class of flower (Iris-setosa, Iris-virginica and Iris-versicolor). Each class is having 50 samples. Our goal is to make unbalance data-set balanced so I have deleted 45 samples of Iris-virginica. Now in new data-set we have only 5 sample of Iris-virginica (minor class).
 

Our goal for this tutorial includes:

1) To multiply samples of Iris-virginica
2) To visualize all sample of Iris-virginica and to ascertain that newly synthesized sample are similar to original Iris-virginica samples.

The core idea behind entire algorithm is very simple

1. Lets say we had a iris data with N = 105 samples.

2. Out of N, we have  a class  Iris-virginica having n = 5 samples, And other two classes having 50 samples each.

3. We need to increase number of samples for class  Iris-virginica (minor class)

4. Each sample in iris data-set having 4 attribute, one sample can be represented in array as [6.3, 3.3, 6.0, 2.5, 'Iris-virginica'] 

5. First we will find out samples (nearest neighbors) which are very similar to given minor sample, lets say we got following nearest neighbors [[6.5, 3.0, 5.8, 2.2, 'Iris-virginica'],[6.3, 2.9, 5.6, 1.8, 'Iris-virginica'],[6.4, 2.8, 5.4, 1.9, 'Iris-virginica']]

6. Out of this nearest neighbor one sample is choose at random let say we choose 

7. [6.5, 3.0, 5.8, 2.2, 'Iris-virginica'] to be the nearest one

8. Calculation: 

    1. One minor sample (m) -                          [6.3, 3.3, 6.0, 2.5, 'Iris-virginica']
    2. Random nearest neighbor (r) -              [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']    
    3. Positive Difference between two (d) -      [0.2, 0.3, 0.2, 0.3]
    4. Random number between 0 and 1 (i) -    [0.2, 0.8, 0.3, 0.5]
    5. Synthetic minor sample =  m + d * i -           [6.34, 3.54, 6.06, 2.65, 'Iris-virginica'] --> New synthetic sample

**Function -1. Below mentioned function perform a sole function to load data**

```python
def loadDataset(filename, numattrs):
    """
    loads data from file
    :param filename:
    :param numattrs: number of column in file, Excluding  class column
    :return:
    """
    csvfile = open(filename, 'r')
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)):
        for y in range(numattrs):
            dataset[x][y] = float(dataset[x][y])
    return dataset
```
**Function -2. Another function we would require is to calculate euclidean distance between attributes of two samples:**

```python
def euclideanDistance(instance1, instance2, length):
    """
    calculate euclidean distance between two
    :param instance1:[6.5, 3.0, 5.8, 2.2]
    :param instance2:[6.3, 2.9, 5.6, 1.8]
    :param length: 4 length of array
    :return:
    """
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
```

If we execute above given function with set of array $[6.5, 3.0, 5.8, 2.2]$ and $[6.3, 2.9, 5.6, 1.8]$ then it will give output as 0.
$euclideanDistance([6.5, 3.0, 5.8, 2.2],[6.3, 2.9, 5.6, 1.8],4)$
Euclidean distance starts from 0.0 (Exactly similar) to any number indicating how far two objects are.

**Function -3. Next function is  getNeighbors.  getNeighbors function will take three inputs**

- Entire data-set 
- A minorsample and 
- Number of neighbors to be found out

The mechanism behind this function is as follows:

1.  For a given minor sample find out euclidean distance with all samples in data-set

2.  Sort all samples in ascending order of euclidean distance so that most similar will come first

3.  From beginning take required  number of neighbors to be found out except first sample ( The first sample will be the same minor sample from entire data-set -  most similar )

```python
def getNeighbors(trainingSet, eachMinorsample, k):
    """
    will give top k neighbors for given minor sample (eachMinorsample) in dataset (trainingSet)
    :param trainingSet:  here entire data-set is a training set
    :param eachMinorsample:
    :param k: number of nearest neighbors to search for each minor sample value, using Euclidean distance
    :return: top k neighbors output: # Minor Sample:  [6.3, 3.3, 6.0, 2.5, 'Iris-virginica'] | Neighbors:  [[6.5, 3.0, 5.8, 2.2, 'Iris-virginica'],
        #    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica'],
    """
    distances = []
    length = len(eachMinorsample) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(eachMinorsample, trainingSet[x], length)  # get euclidean distance for all matches
        distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))  # sort as per distance and get top 3 excluding first one
        neighbors = []
    for x in range(k):
        neighbors.append(distances[x + 1][0])  # X+1 in the sorted list ensure that the minor sample itself is not selected as neighbors
    return neighbors
```

**Function -4. seperateMinority will separates all minor samples as per class specified from data-set. This will return an array of all minor samples.**

```python
def seperateMinority(dataSet, MinorClassName, classColumnNumber):
    """
    will separate given minor class from the entire dataset
    :param dataSet: Entire dataset
    :param MinorClassName:  name of minor class, e.g. MinorClassName = "Iris-virginica"
    :param classColumnNumber: column number where class is present [zero indexed]
    :return:
    """
    minorSamples = []
    for eachSample in dataSet:
        if (eachSample[classColumnNumber] == MinorClassName):
            minorSamples.append(eachSample)
    return minorSamples
```

**Function -5. populate is the main function which takes following parameters:**

1. N:  factor by which sample needs to be increase, e.g. 2 means twice

2. minorSample: all minor samples

3. nnarray: nearest neighbor array   [[2.4, 2.5, 'a'],[2.3, 2.2, 'a'],[2.5, 2.5, 'a']]

4. numattrs: equals to number of feature (3) , in 0 based index it iterates from

This function perform actual SMOTE algorithm on datset. It perform following steps

1. Take minorsample and nearest neighbors. 

2. Find out difference between attributes of minorsample and nearest neighbors as diff

3. Generate a positive random float number between 0  and 1 known as gap

4. New attributes of minorsample =  (attributes of minorsample) + gap * difference

5. As described above, all attribute for a particular sample are generated

6. Such  controlled randomness is added to N number of samples to generate synthetic ones

```python
def populate(N, minorSample, nnarray, numattrs):
    """
    perform actual algorithm
    1) take minorsample and nearest neighbours. 
    2) find out difference between attributes of minorsample and nearest neighbours as diff
    3) generate a positive random float number between 0  and 1 known as gap
    4) new attributes of minorsample =  (attributes of minorsample) + gap * difference
    5) As described above, all attribute for a particular sample are generated
    6) such  controlled randomness is added to N number of samples to generate synthetic ones
    
    :param N:  factor by which sample needs to be increase, e.g. 2 means twice
    :param minorSample: all minor samples
    :param nnarray: nearest neighbour array   [[2.4, 2.5, 'a'],[2.3, 2.2, 'a'],[2.5, 2.5, 'a']]
    :param numattrs: equals to number of feature (3) , in 0 based index it iterates from 0,1,2,3
    :return:
    """
    while (N > 0):
        nn = randint(0, len(nnarray) - 2)
        eachUnit = []
        for attr in range(0, numattrs+1): #[0,1,2,3] iterate over each attribute (feature)
            diff = float(nnarray[nn][attr]) - (minorSample[nn][attr]) # difference between nearest neighbour and actual minor sample
            gap = random.uniform(0, 1) # generate a random number between 0 and 1
            eachUnit.append(minorSample[nn][attr] + gap * diff) # multiply difference with random number and add this to original attribute value
        for each in eachUnit:
            syntheticData.write(str(each)+",")
        syntheticData.write("\\n")
        N = N - 1
```

**Function -6. The last function is SMOTE function , which call all these above functions and serve as an entry point for the work-flow. k is  number of nearest neighbors to be taken in to consideration  to generate synthetic samples.**

```python
def SMOTE(T, N, minorSamples, numattrs, dataSet, k):
    """
    :param T = Number of minority class Samples # here we have 5
    :param k = k mean (clustering value)
    :param minorSample: all minor samples
    :param N = "Number of sample to be generated should be more than 100%"
        Amount of smoted sample required  N%
    """
    if (N <= 100):
        print "Number of sample to be generated should be more than 100%"
        raise ValueError
    N = int(N / 100) * T  # N = number of output samples required
    nnarray = []
    for eachMinor in minorSamples:
        # nnarray all nearest neighbour [[2.4, 2.5, 'a'],[2.3, 2.2, 'a'],[2.5, 2.5, 'a']]
        nnarray = (getNeighbors(dataSet, eachMinor, k))
    populate(N, minorSamples, nnarray, numattrs)
```

I am not yet done, I have generated synthetic data but who knows I am right or not. If this would be 2-D data I would directly plot on graph and can visualize but this is 4-D data, A technique called t-sne (t-Distributed Stochastic Neighbor Embedding ) is used to visualize multi - dimensional data. If you are unaware about t-sne, go through my previous [tutorial](https://www.machinelearningpython.org/single-post/2017/04/25/t-Distributed-Stochastic-Neighbor-Embedding-t-SNE-in-python).

After running file visualize.py I got below given Image:

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_7a84e5123cdb4ffbb3f72b51d6908486~mv2.png/v1/fill/w_681,h_681,al_c,usm_0.66_1.00_0.01/884a24_7a84e5123cdb4ffbb3f72b51d6908486~mv2.png"></p>


<p align="center">Figure. 1 Samples of  Iris-virginica before and after application of SMOTE algorithm</p>

Increase in Dark blue samples clearly represents that we have successfully synthesized sample for  Iris-virginicaI am currency working on making this implement multi threaded, Soon will update this tutorial.
