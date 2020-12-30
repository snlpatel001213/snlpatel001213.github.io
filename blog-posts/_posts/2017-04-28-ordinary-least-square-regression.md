---
layout: post
title: Ordinary Least Square Regression
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-04-28
category: articles
tags: [Statistics]
img: olsr.png
comments: true
share: true
---

You may refer this literature for mathematical explanation of below implemented algorithm 
1) http://www.kenbenoit.net/courses/quant1/Quant1_Week8_OLS.pdf
2) http://www.openaccesstexts.org/pdf/Quant_Chapter_05_ols.pdf
 
All codes discussed here can be  found at my [Github](https://github.com/snlpatel001213/algorithmia/tree/master/regression/ordinaryLeastSquareRegression) repository
 
For effective learning I suggest, you to calmly go through the explanation given below, run the same code from Github and then read mathematical explanation from above given links.
 
Code compatibility : Python 2.7 Only
 
To get this code running execute main.py file as given in GitHub repository

---

Ordinary Least Square Regression (OLSR) is very similar to the procedure describe in tutorial of stochastic gradient. In fact stochastic gradient is one of the part of OLSR in optimization step.
 
So far we have seen two variable in data x any y. where x independent variable and value of y (dependent) depends on x.  But real life data often remain multivariate. There can be many independent variable X:[x1, x2, x3]  and one dependent variable (Y).
 
For single variable problem we can represent OLSR as shown below: 

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_20db69742e7e45fdb7b146673a29d0dc~mv2.png/v1/fill/w_218,h_214,al_c,usm_0.66_1.00_0.01/884a24_20db69742e7e45fdb7b146673a29d0dc~mv2.png"></p>


<p align="center">Figure 1.  An idea of Regression.</p>

As shown above our task is to find a trend-line which lies in such way that distance(shown in green ) between all data-points (red) and trend-line (blue) should be  minimum. case with multivariate system is little complex and can be represented by below given equation. 
$γ = \beta0 + m1X1 + m2X2 + m3X3$ and so on for n variables.
where,
$\beta o$ is a regression coefficient
$m1,m2 ... mn$ are slop for each variable $X1, X2 ..  Xn$ respectively
In present problem we will use "Pima Indians Diabetes Database"

1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1)


In present case based on variable 1 to 8 it is decided that what is the chance of a person being diabetic (class 1) or not (class 0).
I have coded this tutorial in modular way mail algorithm function are  kept in olsr.py and other supporting function like reading file, minmax calculator and others kept in  in dataUtils.py file.

## Step 1

Read the file ion csv and keep it in form of array of floats.
we have  two function in dataUtils.py 
 
1) **loadFromcsv** - to load data fro csv file.
def loadFromcsv(self,fileName):

    ```python
    """
    load a file and conver to 2d python list and return it
    :param fileName: csv file name with absolute path
    Example file  - pima-indians-diabetes.data
    Test the script using following code
    loadDataInstance =  loadData()
    print loadDataInstance.loadFromcsv('pima-indians-diabetes.data')
    e.g. https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data
    :return: 2D arrat [list of [list]]
    e.g. [['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1'], ['1', '85', '66',...]..[]...]
    """
    try:
     data = list(csv.reader(open(fileName)))
     return data
    except:
     print (traceback.print_exc())
    ```
2) **convertDataToFloat** - convert to data to float

    ```python
    def convertDataToFloat(self,dataset):
    """
    loadFromcsv function returns data as list of list of  strings,
    It must be converted to floats for further processing
    code can be tested through below given snippet
    loadDataInstance = loadData()
    dataset = loadDataInstance.loadFromcsv('pima-indians-diabetes.data')
    print loadDataInstance.convertDataToFloat(dataset)
    :param dataset:
    :return: dataset in floats
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = float(row[i])
            return dataset
    ```
## Step 2.  min-max normalization

Min-max makes  attribute data is scaled to fit into a specific range. As we can see in the 'pima-indians-diabetes.data'  there are 8 variable in different range. For effective learning it is very much required to convert all these variables in one range 0-1. 
This can be done using min max normalization. To know about min max normalization please refer this tutorial.
 
We have  two function in dataUtils.py  to accomplish this task.

1) **minMaxCalculator** - will calculate min and max value for each of the eight variables.

    ```python
    def minMaxCalculator(self,dataset):
        """
        :param dataset: data set is expected to be 2D matrix
        :return: minMax ==> list of list, e.g. [[12, 435], [13, 545], [5, 13424], [34, 454], [5, 2343], [4, 343]]
        To run this individual function for testing use below given code
        minMaxNormalizationInstance = minMaxNormalization()
        minMaxNormalizationInstance.minMaxCalculator([[12,13,13424,34,2343,343],[435,545,5,454,5,4],[43,56,67,87,89,8]])
        """
        minMax = list()
        for columnNo in range(len(dataset[0])):
            """
            len(dataset[0]) is number of elements present in the row e.g. columns
            iterating by column
            e.g. this is the column  then, [[12,13,13424,34,2343,343],[435,545,5,454,5,4],[43,56,67,87,89,8]]
            """
            columnvalues = []
            for row in dataset:
                """
                going to each row for particular column
                """
                columnvalues.append(row[columnNo])
            """
            e.g. columnvalues [12, 435, 43] at the end
            """
            minimumInColumn = min(columnvalues)
            maximumInColumn = max(columnvalues)
            """
            this will be the min and max value for first column  12 435
            """
            minMax.append([minimumInColumn,maximumInColumn])
            """
            This will be in minMax list at the end of all iteration on all column
            where each sublist represent min and max value for that column respectively
            [[12, 435], [13, 545], [5, 13424], [34, 454], [5, 2343], [4, 343]]
            """
        return minMax
    ```
    
    For our case this function would return [[0.0, 17.0], [0.0, 199.0], [0.0, 122.0], [0.0, 99.0], [0.0, 846.0], [0.0, 67.1], [0.078, 2.42], [21.0, 81.0], [0.0, 1.0]], where each sub-array e.g. [0.0, 17.0] represents  min and max value for given attribute column.
 
2) **normalizeDatasetUsingMinmax** -  wil use above information about min and max value in each column and by usibg  below given formula it will calculate resize all data between 0 and 1 
    <center>
    
    $$z_i = \frac{x_i - min(x)}{max(x) - min(x)}$$

    Where,

    $X_i$ is any data point in column $x$.

    $min (x)$ is minimum value in column $x$  as calculated above 

    $max (x)$ is maximum value in column $x$  as calculated above 
    </center>


    ```python
    def normalizeDatasetUsingMinmax(self,dataset,minMax):
        """
        Actual implementation of min max normalization where it accepts data set and minmax value for each column
        y= (x-min)/(max-min)
        :param dataset: dataset in 2d array
        :param minMax: [[12, 435], [13, 545], [5, 13424], [34, 454], [5, 2343], [4, 343]]
        :return: will return min max value e.g. [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.07328605200945626, 0.08082706766917293, 0.004620314479469409, 0.1261904761904762, 0.03592814371257485, 0.011799410029498525]]
        This snippet of code can be tested using following code
        minMaxNormalizationInstance = minMaxNormalization()
        dataset = [[12, 13, 13424, 34, 2343, 343], [435, 545, 5, 454, 5, 4], [43, 56, 67, 87, 89, 8]]
        minmax = minMaxNormalizationInstance.minMaxCalculator(dataset)
        print minMaxNormalizationInstance.normalizeDatasetUsingMinmax(dataset, minmax)
        """
        for row in dataset:
            for eachColumnNo in range(len(row)):
                """
                where
                minMax[eachColumnNo][1] = max for the given column
                minMax[eachColumnNo][0] = min for the given column
                """
                row[eachColumnNo] = float((row[eachColumnNo]-minMax[eachColumnNo][0]))/float((minMax[eachColumnNo][1]-minMax[eachColumnNo][0])) # re-assigning minimized value to array : row
        return dataset
    
    ```
    After application of normalizeDatasetUsingMinmax, each row in data-set would look like this (between 0 and 1) : [0.11764705882352941, 0.37185929648241206, 0.0, 0.0, 0.0, 0.0, 0.010247651579846282, 0.016666666666666666, 0.0]
 
## Step 3.
As we have data-set ready, let split it into train and test.
1) **basicSplitter** -  I have designed a basic data splitter, that would devide data into 70%(train) and 30%(test)

    ```python
    def basicSplitter(self,dataset):
        """
        Just take the dataset and split it in to 70% and 30% ration
        :param dataset:
        use following code to run this code snippet
        loadDataInstance = loadData()
        dataset = loadDataInstance.loadFromcsv('pima-indians-diabetes.data')
        dataset =  loadDataInstance.convertDataToFloat(dataset)
        splitToTrainTestInstance =  splitToTrainTest()

        :return: test and train 2d array
        """
        trainDataSize = int(len(dataset)*0.7)
        testDataSize = int(len(dataset) - len(dataset)*0.7)
        print ("Train data size : ",trainDataSize," | Test data size : ",testDataSize)
        train = dataset[:int(len(dataset)*0.7)]
        test = dataset[int(len(dataset) * 0.7):]
        return train, test
    ```

## Step 4. Actual training

Training start by calling these below  function repeatedly and updating coefficient as discussed in stochastic gradient tutorial.
 
1) Predict - used to predict on data using regression algorithm as discussed. It takes two inputs.
    - Data - only X part
    - Previously learned coefficients and slops and predicts on given data.
2) stochastic_gradient - used for to  minimize errors and takes three 3 inputs
    - Data - whole data X and Y
    - learning rate (η) - Its is a value greater than 0 and lesser than or equal to 1 [ 0< η >=1]
    - Epochs - Number of time the same data to be given to the machine learning algorithm so that it can learn.
 
Working of these two function can be nicely explained by below given figure

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_fb621c12d89b4a53aa31ab2ef541a006~mv2.png/v1/fill/w_738,h_421,al_c,lg_1/884a24_fb621c12d89b4a53aa31ab2ef541a006~mv2.png"></p>

 
<p align="center">Figure. 2 Working of prediction and stochastic gradient function</p>
 
```python
def predict(self, Xrow,coefficients):
     """
     for prediction based on given row and coefficients
     :param Xrow:  [3.55565,4.65656,5.454654,1] where last element in a row remains Y [so called actual value y-actual]
     :param coefficients: [0.155,-0.2555,0.5456] previously updated coefficients by stochastic_gradient used for prediction
     :return: Ypredicted
     coefficient  can be actually compared with memory from learning and be applied for further predictions
     """
     Ypredicted = coefficients[0]
     for i in range(len(Xrow)-1):
         Ypredicted += Xrow[i]*coefficients[i+1]
     return Ypredicted
```

Here it is important to note that we have used linear function  - $γ = \beta 0 + m1X1 +m2X2$. 
$Y_predicted = coefficients[0]$  represent addition of $β0$ in the equation.
$Ypredicted += Xrow[i]*coefficients[i+1]$ represent addition of $X_1m_1$ to equation.
 
```python
def stochastic_gradient(self, trainDataset, learningRate, numberOfEpoches):
     """
     :param trainDataset:
     :param learningRate:
     :param numberOfEpoches:
     :return: updated coefficient array
     """
     """
     For each column in train dataset we will be having one coefficient
     if training dataset having 5 column per array than
     coefficient array will be something like this [0.0, 0.0, 0.0, 0.0, 0.0]
     """
     coefficient = [0.1 for i in range(len(trainDataset[0]))]
     for epoch in range(numberOfEpoches):
         """
         for each epoch repeat this operations
         """
         squaredError = 0
         for row in trainDataset:
             """
             for each row calculate following things
             where each row will be like this [3.55565,4.65656,5.454654,1] ==> where last element in a row remains Y [so called actual value y-actual]
             """
             Ypredicted = self.predict(row,coefficient) # sending row and coefficient for prediction
             error = row[-1] - Ypredicted #row[-1] is last elemment of row, can be considered as Yactual; Yactual - Ypredicted gives error
             """Updating squared error for each iteration""""
             squaredError += error**2
             """
             In order to make learning, we should learn from our error
             here  we will use stochastic gradient as a optimization function
             Stochastic gradient for each coefficient [b0,b1,b1,.....] can be formalized as
             coef[i+1]  =  coef[i+1] + learningRate * error * Ypredicted(1.0 - Ypredicted)* X[i]
             For a row containing elements [x1, x2, x3, x4, x5], coefficient  [bo, b1, b2, b3, b4, b5]
               where each coefficient belongs to each element in a row
               e.g. b1 for X1, b2 for x2 and so on..
             As coefficient[i] here is equal to bo, e.g. row element independent, we will update it separately.
             """
             coefficient[0] = coefficient[0]+learningRate*error*Ypredicted*(1+Ypredicted)
             for i in range (len(row)-1):
              coefficient[i+1] = coefficient[i+1] + learningRate * error * Ypredicted*(1.0 - Ypredicted)* row[i]
              """
              lets print everything as to know whether or not the error is really decreasing or not
              """
              print (">>> Epoch : ", epoch ," | Error : ", squaredError)
     return coefficient
```

## Step 5. Testing
We have learned things from training step and stored our learning in form of coefficients. We had eight attributes in our data-set so by equation  γ = β0 + m1X1 + m2X2 + .. + m8X8 we should have Nine coefficient [ β0 and m1 to m8] as shown below.

Coefficient = [-0.40134710602161927, 0.28245197540248934, 0.7001137271400164, -0.1876675370110126, 0.0022099414467335937, 0.11226076950136779, 0.36010329497830973, 0.25824794282244273, 0.1412229577339494]
For the test dataset we will use the same equation here γ is unknown to us so by applyingwe will find out
y. β0 + m1X1 + m2X2 + .. + m8X8.

```python
def predictOnTest(self,testDataset, coefficient):
     """
     predicting on leftover dataset
     :param testDataset: test dataset Array of array
     :param coefficient: Array of coefficient
     :return: actualvalues [Array],predictedlist [Array]
     """
     actual  = [] # stores actual value of Y
     predictedlist = [] # stores predicted  value of Y
     for row in testDataset:
         actual.append(row[-1])
         predicted = coefficient[0]
         for i in range(len(row)-1):
             predicted +=row[i]*coefficient[i+1]
         predictedlist.append(predicted)
         print ("predicted : ",predicted, " | Actual : ",row[-1] )
     return actual,predictedlist
```

## Step 6. Getting performance matrix 

If you look into predictedlist returned by  predictOnTest function you would find  float numbers like this : [0.004066216633807906, 0.23066261620267448, 0.30185797638280926, 0.3264935294499418, 0.25377601414111306, 0.30344480864150913, 0.060468595846753875, 0.037823825442416865, 0.5389956781340269, 0.665507821872288, ,,,].
These are not our classes. To get classes (diabetic /Non-diabetic  ) we need to decide certain threshold, if predicted value is above it it will be 1 else 0. In ths way will get classes. Same thing as discussed I have implemented with below given createConfusionMatrix function.
If you are unaware about performace measure in binary classification, read  Performance Measures part of my earlier blog post.

```python
def createConfusionMatrix(self,actual, predicted, threshold):
     """
     will create confusion matrix for given set of actual and predicted array
     :param actual: Array of Actual sample
     :param predicted: Array of predicted sample
     :param threshold:  Any number between 0-1
     :return:
     """
     fp = 0
     fn = 0
     tp = 0
     tn = 0
     for i in range(len(predicted)):
         if predicted[i] > threshold:
             predicted[i] = 1
         else:
             predicted[i] = 0
     for no in range(0, len(predicted)):
         if predicted[no] == 1 and actual[no] == 1:
             tp += 1
         elif predicted[no] == 0 and actual[no] == 0:
             tn += 1
         elif predicted[no] == 1 and actual[no] == 0:
             fn += 1
         elif predicted[no] == 0 and actual[no] == 1:
             fp += 1
     ACC = float((tp + tn))/ float((fp + tp + tn + fn))
     F1 = float(2*tp)/ float(2*tp + fp + fn)
     print "False Positive : ",fp,", False Negative : ", fn,", True Positive : ", tp,", True Negative : ", tn,", Accuracy : ",ACC ,", F1 Score : ",F1
 
Here for this tutorial I have fixed the thresholf to 0.30, and I got following performance measures : 
False Positive :  26 , False Negative :  23 , True Positive :  53 , True Negative :  129 , Accuracy :  0.787878787879 , F1 Score :  0.683870967742
 
Overall process of Ordinary Least Square Regression can be summarized as given in below flowchart:
```

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_289fa5671d3f4edfac731872f7867222~mv2.png/v1/fill/w_577,h_676,al_c,usm_0.66_1.00_0.01/884a24_289fa5671d3f4edfac731872f7867222~mv2.png"></p>


<p align="center">Figure 3. Functional overview of ordinary Least Square Regression</p>
 
