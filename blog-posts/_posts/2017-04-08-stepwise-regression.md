---
layout: post
title: Stepwise Regression
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-04-08
category: articles
tags: [Statistics]
img: stepwise_regression.webp
comments: true
share: true
---

You may refer this literature for mathematical explanation of below implemented algorithm 
1) http://statweb.stanford.edu/~ckirby/lai/pubs/2009_Stepwise.pdf

All codes discussed here can be  found at my [Github](https://github.com/snlpatel001213/algorithmia/tree/master/regression/stepwiseRegression) repository
 
For effective learning I suggest, you to calmly go through the explanation given below, run the same code from Github and then read mathematical explanation from above given links.
 
Code compatibility : Python 2.7 Only
To get this code running run main.py file as given in GitHub repository

---
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_3d8beedc567b4dc2a437e43b11b9f6e4~mv2.jpg/v1/fill/w_690,h_404,al_c,q_80,usm_0.66_1.00_0.01/884a24_3d8beedc567b4dc2a437e43b11b9f6e4~mv2.webp"></p>

<p align="center">Figure 1. Few of the many factors affecting Home Price in Boston </p>
 
Step Wise Regression is all about selecting attributes those helps best in prediction.
We can have large number of attributes [x1, x2, x3 ,,,,,x100] and out of these only few are actually contributing to outcome (Y). 
How to choose these variables then?. This can be done with Step Wise Regression. In Step Wise Regression we will separately use each attribute column to predict on Y and then all those attribute columns will be selected having minimum Squared errors.
For present tutorial I have selected house price data-set [insert hyperlink]. This data is about factor affecting housing price in Boston.
In entire data-set there are different values for 13  factors (attributes) are provided. Here we are suppose to predict MEDV  (Median value of owner-occupied homes in $1000's) 

1. CRIM      per capita crime rate by town
2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS     proportion of non-retail business acres per town
4. CHAS      Charles River dummy variable (= 1 if tract bounds  river; 0 otherwise)
5. NOX       nitric oxides concentration (parts per 10 million)
6. RM        average number of rooms per dwelling
7. AGE       proportion of owner-occupied units built prior to 1940
8. DIS       weighted distances to five Boston employment centres
9. RAD       index of accessibility to radial highways
10. TAX      full-value property-tax rate per $10,000
11. PTRATIO  pupil-teacher ratio by town
12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks  by town
13. LSTAT    % lower status of the population
14. MEDV     Median value of owner-occupied homes in $1000's

**Workflow**

Our job is to find those attributes out of above given 13 represent the data or affect (house price) the most.
Here is the work flow of entire implementation.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c01cfedd365547fa92f41450280ccee5~mv2.png/v1/fill/w_596,h_488,al_c,usm_0.66_1.00_0.01/884a24_c01cfedd365547fa92f41450280ccee5~mv2.png"></p>

<p align="center">Figure 2. Step Wise Regression working implementation</p>
 
As shown in figure we will be mainly going through following steps:
1) Take a data-set, we already took :)
2) Apply min max normalization for more information regarding min-max normalization, refers this tutorial.
3) Separate X and Y from data-set with function 

    ```python
    def getXandY(self,dataset):
        """
        To seperate predictor and predicate from given dataset
        :param dataset: 2D array
        :return: predictors : X and predicate : y
        """
        x =[]
        y=[]Github
        for row in dataset:
            x.append(row[:-1]) # all X
            y.append(row[-1]) # last column is Y
        return x, y
    ```

4)  stepwise_Regression function will pass each column (attribute) to the giveErrorForColumn function and collect error for each column separately. It will take following parameters.
 
    X : columns (attribute
    Y : predicate
    learningRate (η) : Its is a value greater than 0 and lesser than or equal to 1 $[ 0< η >=1]$
    numberOfEpoches : Number of time the same data to be given to the machine learning algorithm so that it can learn.
    MaxPredictor : Number of most relevant column to be selected, It should be more than 0 and less than max(attributes)
 
    ```python
    
    def stepwise_Regression(self, X, Y, learningRate, numberOfEpoches, maxPredictor):
        """
        :param Xtrain: 2D array[[-0.04315767605226367, 1.165157142272148, -0.45268840110500463,,,],[-0.04315767605226367, 1.165157142272148, -0.45268840110500463,,,],[],[]]
        :param Ytrain: 2d Array [[.45],[.67],[.46],[.4432]]
        :param learningRate: float between 0 and 1
        :param numberOfEpoches: any integer
        :param maxPredictor: number of column to be selected for final run with highest correlation with predicate
        :return: acceptedColsNo,coefficientValue
        e.g.
        (7.5460202999842885, [-0.04315767605226367, 1.165157142272148, -0.45268840110500463, -0.43397502762968604, -0.7226568412142056])
        """
        """
        For each column in train dataset we will be having one coefficient
        if training dataset having 5 column per array than
        coefficient array will be something like this [0.0, 0.0, 0.0, 0.0, 0.0]
        """
        allSquaredError = []
        coefficientValue = []
        for columnNo in range(0, len(X[0])):
            "getting squared error for each column"
            XColumn = loadDataInstance.getSpecificColumns(X, [columnNo])  # getting data for each column
            squaredError, coefficient =self.giveErrorForColumn(XColumn, Y, learningRate, numberOfEpoches) # passing to estimate error for given column w.r.t. output column
            allSquaredError.append(squaredError) # storing squared error for each column
            coefficientValue.append(coefficient) # stroing coefficient for each column
        print "Squared Error : ",allSquaredError
        "getting maxcolumn e.g. maxPredictor with lowerst error"
        acceptedColsNo = self.accepetedColumns(allSquaredError, maxPredictor)
        "Making prediction on selected column"
        return acceptedColsNo,coefficientValue
    ```
5) giveErrorForColumn - It is the same stochastic gradient descent function we saw in the earlier tutorial. We will be using the same here to minimise error and return back Squared error.It takes following attributes:
 
   Xcolumn : columns (attribute)
   Y : predicate
   learningRate (η) : Its is a value greater than 0 and lesser than or equal to 1 [ 0< η >=1]
   numberOfEpoches : Number of time the same data to be given to the machine learning algorithm so that it can learn.

   ```python
   
    def giveErrorForColumn(self, Xcolumn, Y, learningRate, numberOfEpoches):
            """
            :param Xtrain:
            :param Ytrain:
            :param learningRate:
            :param numberOfEpoches:
            :return:
            """
            coefficient = [0.1 for i in range(len(Xcolumn[0]) + 1)] # initializing coefficient
            # print coefficient
            for epoch in range(numberOfEpoches):
                """
                for each epoch repeat this operations
                """
                squaredError = 0
                # print X
                for rowNo in range(len(Xcolumn)): #
                    """
                    for each row calculate following things
                    where each row will be like this [3.55565,4.65656,5.454654,1] ==> where last element in a row remains Y [so called actual value y-actual]
                    """
                    # print Xtrain[rowNo]
                    Ypredicted = self.predict(Xcolumn[rowNo], coefficient)  # sending row and coefficient for prediction
                    """
                    row[-1] is last elemment of row, can be considered as Yactual; Yactual - Ypredicted gives error
                    """
                    error = Y[rowNo] - Ypredicted
                    "Updating squared error for each iteration"
                    squaredError += error ** 2
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
                    #updating Bo
                    coefficient[0] = coefficient[0] + learningRate * error * Ypredicted * (1 + Ypredicted)
                    #moving in to entire column
                    for i in range(len(Xcolumn[rowNo])):
                        #updating Bn, where n can be any number from 1 to number of attribute(N)
                        coefficient[i + 1] = coefficient[i + 1] + learningRate * error * Ypredicted * (1.0 - Ypredicted) * \
                                                                Xcolumn[rowNo][i]
                        """
                        lets print everything as to know whether or not the error is really decreasing or not
                        """
                # print (">>> Epoch : ", epoch, " | Error : ", squaredError, "| Average Error : ", averageError)
            return squaredError,coefficient
    ```

6) predict - giveErrorForColumn function will call predict function with following parameters

    Xrow: given value of  row in column 

    coefficients :  $B_o and B_n$, where n can be any number from 1 to number of attribute(N)

    ```python
    
    def predict(self, Xrow, coefficients):
            """
            for prediction based on given row and coefficients
            :param Xrow:  [3.55565,4.65656,5.454654,1] where last element in a row remains Y [so called actual value y-actual]
            :param coefficients: [0.155,-0.2555,0.5456] Random initialization
            :return: Ypredicted
            This function will return coefficient as it is real thing we get after training
            coefficient  can be actually compared with memory from learning and be applied for further predictions
            """
            Ypredicted = coefficients[0]
            for i in range(len(Xrow)):
                Ypredicted += float(Xrow[i]) * float(coefficients[i + 1])
            return 1.0 / (1.0 + exp(-Ypredicted))
    ```

7) accepetedColumns – It takes squared error for all the columns and column numbers with minimum error will be returned. this takes following parameters

    squaredError  - Squered errors for all attributes
    [19.870694243392535, 18.49064887825751, 16.173082492771197, 20.453729207679526, 17.349173281094615, 15.891363738470233, 18.141040302112437, 19.951988433235435, 18.044399089376622, 16.43915654444947, 15.89859is acce1777034301, 19.20710063941564, 10.42727645916291]

    maxPredictor – max number of columns with minimum squared error to be returned

    ```python
    
    def accepetedColumns(self,squaredError,maxPredictor):
            """
            get columns with least r square value
            :param squaredError: An array with R square value
            e.g. [12.399790911951232, 11.14467573048574, 10.99939946366844, 12.313118044897763, 11.812905343161896, 8.530664160073936, 11.642709319002446, 12.377547637064676, 12.376152652375172, 11.935468510009718, 10.221164713630898, 12.258299424118913, 8.627925610329616]
            :param maxPredictor: An integer indicating number of column out of these to be selected e.g. 4
            :return: selected column number [5, 11, 9, 2] A list
            """
            acceptedColsNo = []
            for i in range(maxPredictor):
                minValue = min(squaredError)
                acceptedColsNo.append(squaredError.index(minValue))
                squaredError.remove(minValue)
            return acceptedColsNo
    ```

After running main.py file you will get following output : [12, 5, 9, 2, 7]. I have selected data for all  [12, 5, 9, 2, 7] columns and plotted against median house price, I got the plot given below.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c01cfedd365547fa92f41450280ccee5~mv2.png/v1/fill/w_596,h_488,al_c,usm_0.66_1.00_0.01/884a24_c01cfedd365547fa92f41450280ccee5~mv2.png"></p>

<p align="center">Figure 3. Selected Columns by Step Wise Regression</p>
 
Step wise regression can be used very well for feature(attribute) selection, that affects our class the most. There is only one drawback of this method is id we cannot use it for categorical variables.
There are other kind of step-wise regression where nature of activation function is changed and whichever activation function gives best result is accepted. We will see one example for this in future.
In future we will see more methods like Principle Component Analysis which are used for feature space minimization.
 
