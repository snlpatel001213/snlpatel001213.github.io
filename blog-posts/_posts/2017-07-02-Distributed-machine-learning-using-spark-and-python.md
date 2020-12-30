---
layout: post
title: Distributed machine learning using sparkxc
date: 2017-07-02 12:54:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
tag: [Spark, H2O, Python]
img: h2o.png
comments: true
share: true
---

H2o (h2o.ai), developed a really cool and lucid way to integrate machine learning with any application. When I begun learning machine-learning 3 years before, I started it with h2o. h2o core engine is JVM but provides api fir python, R, scala. h2o goes well with single machine but It also supports distributed computing on spark. 
 
Although standalone mode runs fine, it creates problem with large data-set when model size goes beyond 256mb.  As per h2o memory requirement it is said that, h2o require 4 times memory then size of data-set for optimum performance. For above said reason its better to deploy h2o on spark and use it by whichever API you want.
 
In present tutorial we will deploy  sparking water (a version of h2o compatible to spark)  on cluster running spark and will run machine learning algorithms on spark cluster with python.
 
Here I have two machines, a local machine and a remote server with ip address 192.168.40.40
 
Before doing anything we will download required packages

**Download Required packages**
1) Download spark from  -  http://spark.apache.org/downloads.html
   <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_b4a85d39d1014ad585c504178302c85a~mv2.png/v1/fill/w_794,h_407,al_c,usm_0.66_1.00_0.01/884a24_b4a85d39d1014ad585c504178302c85a~mv2.png">
 Choose spark release and package type according to given screenshot :

  2) Download sparkling water from -  http://h2o-release.s3.amazonaws.com/sparkling-water/rel-1.5/6/index.html
 
Upon downloading 1 and 2, transfer those to remote machine (server) and un-zip/un-tar both packages in /home/ directory.
 
**Setting spark path on server**

1. go to `/home` 
2. issue a command sudo nano .bashrc
3. add following lines to end of the file to set spark path
`export SPARK_HOME=/home/$USER/spark-2.0.0-bin-hadoop2.6`
`PATH=${SPARK_HOME}/bin:${PATH}`
`export PATH`
 
**Running h2o on spark**

1.  go to `/home/$USER/sparkling-water-2.0.0/bin`
2. Issue following commands
we will make cluster of 1 slave and 2 core, for ease of understanding. You can go as many slave as you have in spark cluster practically.
 
    `export MASTER="local-cluster[1,2,1024]" 
sparkling-shell --conf "spark.executor.memory=1g"`
 
    You will see following output : 

    <p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_2c5158715ad448a1af7d78ba926a6b9b~mv2.png/v1/fill/w_769,h_542,al_c,lg_1/884a24_2c5158715ad448a1af7d78ba926a6b9b~mv2.png">

This will leave you to a scala shell, where you need to fire following commands in order to form a h2o cloud.

```
import org.apache.spark.h2o._
val h2oContext = new H2OContext(sc).start() 
import h2oContext._ 
```

You will see following output for 3 of the above commands.
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_3a36b5d786204c0b8988506947181ddd~mv2.png/v1/fill/w_763,h_487,al_c,lg_1/884a24_3a36b5d786204c0b8988506947181ddd~mv2.png">

Now h2o is up and running at 192.168.40.40:54323 
 
1) Connecting to h2o at 192.168.40.40:54323  using python From local machine
For to handel preprocessing locally h2o require a local version of it, so we will first install h2o locally.
To install h2o locally, use following command.

`pip install h2o==3.10.0.7`

Upon installation, make a file called temp.py and copy paste following line in to it.

```python
import h2o
h2o.connect(ip="192.168.40.40", port=54323)
```

You will see following output if got connected successfully
```bash
/home/XYZ_us/anaconda2/bin/python /home/user1/PycharmProjects/CAS/transKnock/temp.py
Connecting to H2O server at http://192.168.40.40:54323... successful.
--------------------------  --------------------------------
H2O cluster uptime:         1 hour 22 mins
H2O cluster version:        3.10.0.7
H2O cluster version age:    1 month and 25 days
H2O cluster name:           sparkling-water-sunil_-750106481
H2O cluster total nodes:    1
H2O cluster free memory:    401.7 Mb
H2O cluster total cores:    24
H2O cluster allowed cores:  24
H2O cluster status:         locked, healthy
H2O connection url:         http://192.168.40.40:54323
H2O connection proxy:
Python version:             2.7.11 final
--------------------------  --------------------------------
Process finished with exit code 0
```

Add few more lines to the temp.py. (we are applying deep learning with 3 hidden layers of size 10,10,10 and looking at the performance)
For the same purpose you will require iris.txt file  which you can get it from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
 
```python
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
trainFile = "iris.txt" 
trainFrame = h2o.upload_file(trainFile, header=-1, sep=",")
r = trainFrame.runif(seed=13)  # Random UNIForm numbers, one per row
train = h2o.upload_file(trainFile)
splits = train.split_frame(ratios=[0.75], seed=1234)
dl = H2ODeepLearningEstimator(activation="Tanh", hidden=[10, 10, 10], epochs=10, seed=13,
                           hidden_dropout_ratios=[0.5,0.5,0.5])
dl.train(x=[0,1,2,3], y=4, training_frame=splits[0])
predictedProabability =  dl.predict(splits[1])
probabilityList = h2o.as_list(predictedProabability, use_pandas=False)
print probabilityList
```

Here we are done with the tutorial!! If you get any error please comment, I will be more than happy to solve it.
