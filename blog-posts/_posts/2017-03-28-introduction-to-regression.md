---
layout: post
title: Introduction to Regression
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-03-28
category: articles
tags: [statistics]
img: introduction_to_regression.png
comments: true
share: true
---

Regression is the process of estimating relationship between dependent variable and one or more independent variables.
Regression makes us to understand effect  independent variables on dependent variable.

[All data / code used for this tutorial is available at my [Github](https://github.com/snlpatel001213/algorithmia/tree/master/regression/IntroductionToRegression) page

To understand Regression, lets start with one example :  
Below given data showing relation between Mortgage interest rates and median home prices.
"Mortgage is a legal agreement by which a bank, building society, etc. lends money at interest in exchange for taking title of the debtor's property, with the condition that the conveyance of title becomes void upon the payment of the debt."

Here we have Home price ($X$) is an independent variable and Mortgage Interest Rate ($Y$) is a dependent variable on $X$. Our goal is to predict value of $Y$  for given value of $X$  by modelling an equation : $y =  mx + b$ .

​​<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_2232936d92a440f48c3a73fdd40a4c46~mv2.png/v1/fill/w_482,h_462,al_c,usm_0.66_1.00_0.01/884a24_2232936d92a440f48c3a73fdd40a4c46~mv2.png"></p>


<p align="center">Figure 1. Overview of data-set.</p>

Below given graph shows negative relation between Mortgage interest rates and median home prices. If we can somehow model this situation then we can predict what would be Mortgage interest rate for home of given price.
​​
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_e60314c936234f4298aa0dd7cc535f4a~mv2.png/v1/fill/w_681,h_383,al_c,lg_1/884a24_e60314c936234f4298aa0dd7cc535f4a~mv2.png"></p>

<p align="center">Figure 2. Regression Line (in orange) passing through entire data (in blue)</p>

To model this situation, the simplest approach would be to make a straight line through the entire data-set (As shown by Orange line). This line is known as trend-line. Any point on trend-line, what would be the Mortgage interest rate  for given house price. 

The trend-line is shown by an equation - $y =  mx + b$ 
Where $m = slop$ ; y = dependent variable(Mortgage interest rate) ;  x =  independent variable(house price) ; b =  regression coefficient

We use two parameter to represent trend-line:
Regression coefficient (b) 
Slop (m)
Now we will see, how to calculate trend-line:
Below given are the equations to calculate trend-line :

<center>

$\bar{Y} = m\bar{X} +c$

$\bar{X} = 1/n*\sum_{i=1}^n xi$ (the average of $x$)

$\bar{Y} = 1/n*\sum_{i=1}^n yi$ (the average of $y$)

$m=\frac{\sum_{i=1}^n(x-\bar{X})(y-\bar{Y})}{\sum_{i=1}^n(x-\bar{X})^2}$

$b = \bar{Y} - m\bar{X}$

​​
Equations for regression, to find slop and regression coefficient.
</center>


I have calculated the same in Microsoft Excel to facilitate understanding. All equations used in excel sheet are as shown above. We are getting the same result here too.
​​
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_be85409588334d3e9147009d1d1b101f~mv2.png/v1/fill/w_719,h_372,al_c,usm_0.66_1.00_0.01/884a24_be85409588334d3e9147009d1d1b101f~mv2.png"></p>

<p align="center">Figure 3. A way to calculate regression in Excel spread sheet.</p>

Similarly we can calculate the same in python using following code : (try it yourself - you will get the same  result)

```python
import math
def findTrendline(xArray,yArray):
    """
    used to find trend line
    Need certain changes in input
    :param XY:
    :return:
    """
    print xArray
    print yArray
    # calculating average for X
    xAvg = float(sum(xArray)) / len(xArray)
    # calculating average for Y
    yAvg = float(sum(yArray)) / len(yArray)
    upperPart = 0.0 # initializing numerator of the slop equation
    lowerPart = 0.0 # initializing denominator of the slop equation
    m = 0.0 # initializing slop
    for i in range(0, len(xArray)):
        #calculating numerator
        upperPart += (xArray[i] - xAvg) * (yArray[i] - yAvg)
        #calculating denominator
        lowerPart += math.pow(xArray[i] - xAvg,2)
    # calculating slop
    m = upperPart / lowerPart
    # calculating regression coefficient
    b = yAvg - m * xAvg
    return m, b
# Example
x =[183800,183200,174900,173500,172900,173200,173200,169700,174500,177900,188100,203200,230200,258200,309800,329800]
y = [10.30,10.30,10.10,9.30,8.40,7.30,8.40,7.90,7.60,7.60,6.90,7.40,8.10,7.00,6.50,5.80]
print findTrendline(x,y)
```

**Pros :** 
- Vary simple techniques
- Good for quick and dirty estimation
- Provide good estimation for linear data

**Cons:**
- Cannot model non linear data; 99.99 % of data are non linear in nature.
