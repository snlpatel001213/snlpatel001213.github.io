---
layout: post
title: Face to Age Prediction Using Convolution Networks
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-05-25
category: articles
tags: [keras, GPU, Python, Machine learning Advance]
img: face2age.png
comments: true
share: true
---
-------
1.  Codes discussed in this blog post can be found at my [GITHUB](https://github.com/snlpatel001213/algorithmia/tree/master/convnet/face2agePrediciton) repository
 
2.  You may download [trained model](https://drive.google.com/file/d/0B5mEsS-c9HHAQkRuLUVvT1VNdFU/view?usp=sharing) for to quickly run face to age prediction or to fine tune on your custom data-set.
 
3.  If you encounter any error, please try to check with [requirements.txt](https://github.com/snlpatel001213/algorithmia/blob/master/convnet/face2agePrediciton/requirements.txt) for proper required python package versions.
 
4.  Code Compatibility : python 2.7 , tested on ubuntu 16.04 with theano as backend
------

In recent days Microsoft is advertising face to age prediction to show azure’s machine learning capability. A website [https://how-old.net/](https://how-old.net/) by Microsoft allows user to upload photo for free and predicts age for the same. Behind the scene a convolution network is working to produce this magic.

In present blog I will be reverse engineering, the science of machine learning behind this technology. The aim of this tutorial is to provide basic implementation of face to age technique. This tutorial is not for any monitory purpose.

# Data Collection #

We require well annotated data-set wherein each photo is tagged with real age. Such data-sets are available at [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). This website is open source collection of images from Wikipedia and IMDB. Each image is having year of birth and time stamp when it was uploaded to  Wikipedia or IMDB. The total collection is about 300GB of data. Looking at my convenience and processing infrastructure I have, I have selected “IMDB Only face data-set” which is of size 7GB in compressed form.

# Data Pre-Processing #

Downloaded images are of various size and formats. For machine learning images need to be of proper dimension and format. 
Each image in data-set is named in a format as shown in example below:

- nm7153885_rm3814127104_1990-8-15_2015.jpg
- nm7153885_rm4089047552_1990-8-15_2015.jpg
- nm7153885_rm4149671424_1990-8-15_2015.jpg
- nm7153885_rm4172933632_1990-8-15_2015.jpg
- nm7153885_rm4206488064_1990-8-15_2015.jpg

Where in the first image, 1990-8-15 is the date of birth (DOB) and 2015 is year of image upload. 
We will choose year from date of birth (DOB)  and difference between date of upload and  year of birth will provide age of image. This can be simply done using regular expression – nm*\\d+_rm\\d+_(\\d+)-\\d+-\\d+_(\\d+).jpg  where for above given example, (\\d+) will capture 1990-8-15 [DOB] and(\\d+) will capture 2015 [year of photo uploaded].

```python
def imageResize(basename,imageName):
	"""
	resize image
	basename : eg. /home/username/XYZFolder
	image name : xyz.jpg
	New folder in the working directory will be created with '_resized' as suffix
	"""
	new_width  = 128
	new_height = 128
	try:  
		img = Image.open(basename+"/"+imageName) # image extension *.png,*.jpg
		img = img.resize((new_width, new_height), Image.ANTIALIAS)
		img.save(basename+'_resized/'+imageName)
	except:
		os.mkdir(basename+'_resized/')
		img = Image.open(basename+"/"+imageName) # image extension *.png,*.jpg
		img = img.resize((new_width, new_height), Image.ANTIALIAS)
		img.save(basename+'_resized/'+imageName)

def resizer(folderPath):
	"""
	to resize all files present in a folder
	resizer('/home/username/XYZFolder')
	"""

	for subdir, dirs, files in os.walk(folderPath):
		for fileName in files:
			try:
				#  print os.path.join(subdir, file)
				filepath = subdir + os.sep + fileName
				#  print filepath
				if filepath.endswith(".jpg" or ".jpeg" or ".png" or ".gif"):
					imageResize(subdir,fileName)
			except:
				print traceback.print_exc()
    # to resize all images in given folder, run below given line

resizer('imdb_crop')
```

# Network Definition #

We will be using VGG16 network architecture, there is no particular reason for using it but stochastically  it does perform well on majority of image recognition use cases.
VGG16 network is as shown in below given image. Output layer is changed to 100, which is equal to the number of classes we possess.

```python
# defining convolutional network
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,128,128)))
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
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(4090, activation='relu'))
model.add(Dense(4090, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))
# model.summary()
```

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png/v1/fill/w_658,h_386,al_c,lg_1/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png"></p>

<p align="center">Figure 1. Architecture of VGG16 Convolution Network used for face to age prediction.</p>

# Training #

IMDB data-set is about 4,50,000 + images and it is very much impractical to keep all of them in memory. So unlike previous tutorial, we will use data generator in this tutorial. Data generator will dynamically read  10,000 images from disk and load it in to RAM. After generating Numpy array and corresponding age vector of all images, data-generator passes 50 images at a time to GPU for processing. VGG16 is a huge network and my GPU cannot accommodate more than 10 images at a time in memory. One can load higher number of images to GPU as per resource availability. 

Provide GENERATOR functioning flow here. 

```python
# this function will load images iteratively in memory
# CPU and GPU memory friendly iterator

def myGeneratorEX(samples_per_epoch,epoch):
    """
    samples_per_epoch : number of images to be loaded in CPU memory at a time
    epoch : number of epochs for training 
    """
    # defining optimizer function
    sgd = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=True)
    # compiling model
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    folderName = "imdb_crop_resized" # folder name where resized images are placed

    fileNames =  glob.glob(folderName+"/*.jpg") #All file names with .jpg extension

    # first 100 imageswill be ued for onthe fly visual performace checking at each iteration
    initialFileNames = fileNames[:100]

    k =0 
    while k < epoch: # for each epoch do following
        print "Epoch : ",k," | Total Images : ",len(fileNames)
        for i in range(len(fileNames)/samples_per_epoch): 
            #All files (~438189) are loaded in memory with batch of size  'samples_per_epoch' e.g.1000
            try:
				# loaded images are converted to numpy array
				x_batch,y_batch = turnToNumpy(fileNames[i\*samples_per_epoch:(i+1)\*samples_per_epoch])

				# such all images are made up of numpy array of range integer 0 - 255(8 bit image)
				# all images are normalised between 0-1 float
				x_batch = x_batch/255.0
				# to check wheather or not our algorithm is learning. to cheack wheather our algorith started differentiating between age.
				x_batch_test,y_batch_test = turnToNumpy(initialFileNames)
				x_batch_test = x_batch_test/255.0
				# fit the data on model
				model.fit(x_batch,y_batch,batch_size=50,nb_epoch=1, verbose=1,validation_split=0.2)
				# test on initial 100 files at each iteration
				test_output = model.predict_classes(x_batch_test)
				print test_output
            except IndexError:
				print traceback.print_exc()
        k = k+1
```

Training takes very long time, On AWS server with 3000+ CUDA core and 11+ GB memory, it took 8 days for me.
I have choose a set of 100 files (online test set) initially and after each iteration i have tested model on online test set to get predicted age. The learning behavior of algorithm is quite intuitive. Initially the algorithm has no clue but after some iteration it starts making sense out of data.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_dc9fdb8552d24008a89e97c10f710edd~mv2.png/v1/fill/w_938,h_542,al_c/884a24_dc9fdb8552d24008a89e97c10f710edd~mv2.png"></p>

<p align="center">Figure 2. Learning progress as training iteration passes. After 468 iteration network actually started learning age difference.</p>

All lines with different color represent different iteration.  At iteration 1 for all 100 online test set images predicted age was 25 years. Similarly for iteration 55, age was 38 for all images.
up to iteration 469 machine had no clue about data but after iteration 469 it slowly started distinguishing between ages by looking at image. At iteration 624 it started predicting in narrow range of 25 to 40. As learning progresses this learning become more powerful and it really start predicting well and in broader range 0 - 100.

# Analysis #

In  present experiment  I have not separated data as test and train. It is very difficult for a machine to remember data from such huge data-set so we can randomly pick few images from train data itself and allow model to predict on the same.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_93279879ffc74d66908d90dd521b9f9e~mv2.png/v1/fill/w_945,h_505,al_c,usm_0.66_1.00_0.01/884a24_93279879ffc74d66908d90dd521b9f9e~mv2.png"></p>

<p align="center">Figure 3. predicted v/s actual age at the end of training.</p>

I have provided a line plot showing result for 500 images with actual and predicted age. It shows following characteristics of learning : 

1. model is capable of predicting extremes of ages
2. with very low data-set 7GB compared to 300GB actual one, model is still doing well

We will be looking at  some samples with their actual and predicted age labelled. 

A. Below are the few images of good predictions.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_027460cc62514e63aa1577c6833c8d58~mv2.png/v1/fill/w_586,h_583,al_c,lg_1/884a24_027460cc62514e63aa1577c6833c8d58~mv2.png"></p>

<p align="center">Figure 4. Good face to age predictions by algorithm [A = Actual, P = Predicted]</p>

Our algorithm predicted well on cases:

1. Where there was an error from database, labeled age was wrong but algorithm predicted it correctly

2. Where algorithm predicted the age correctly, with minor fluctuations.

B. Below are the few images of bad predictions.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_ccdb60ca74284cf4a74900ebc16896ae~mv2.png/v1/fill/w_605,h_446,al_c,lg_1/884a24_ccdb60ca74284cf4a74900ebc16896ae~mv2.png"></p>

<p align="center">Figure 5. These are cases where algorithm performed very badly. [A = Actual, P = Predicted] </p>

C. Effect of multiple faces.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_11a06fbcdac84bc180787971566b3d7e~mv2.png/v1/fill/w_605,h_454,al_c,lg_1/884a24_11a06fbcdac84bc180787971566b3d7e~mv2.png"></p>

<p align="center">Figure 6. Errors in Prediction  due to multiple faces. [A = Actual, P = Predicted]</p>

As I am not using any intermediate face detection and isolation step, multiple faces greatly affects  predictions. For example image no 10 , where Actual age was 8 [of one of the child] but predicted age is 70 [seems correct for aged person].

**Possible Improvements:**

1. Using face detection, this will allow us to better deal with photos with multiple faces

2. Using larger data-sets. I have a gut feeling that Microsoft actually using entire 300 GB of data to give state of the art results
3. Having separate male and female data-set can improve predictions a lot, because apparent features for male and female of same age are different.
