---
layout: post
title: "Object Detection"
img: objectDetection.webp # Add image post (optional)
date: 2017-07-9 12:54:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
tag: [keras, GPU, Python, Machine Learning Advance]
comments: true
share: true
---

All codes discusses in this blog post are placed at [GitHub](https://github.com/snlpatel001213/algorithmia/tree/master/convnet/objectDetect) repository

In case of any error refer to requirements.txt file for python package compatibility or comment  below.

Code Compatibility : python 2.7 , tested on ubuntu 16.04 with theano as backend

Object detection is an application of computational vision related to identifying definite number of objects (such as car, cat or human) from image and video in the given scope of study. Object detection is broader topic which also includes face detection pedestrian detection, physical intrusion detection, and motion detection. Object detection has great applications including in image retrieval and video surveillance.

While searching for older techniques of object detection, I came across this page of Wikipedia describing “Techniques and algorithms” of object detection.

“The advantage we are having is, an image is made of pixels. So in most cases we know the location of next point, it will be connected to our current pixel. Starting with circles, take an image, convert it to grey scale, and detect edges. Move along edges, draw normal, they will intersect at center. Do this for entire circle or find connected edges and calculate Euclidean distance between center and connected points. Another algorithm is move along connected edges rotation of tangent will be uniform, because of symmetry. So whenever there is an abrupt change in rotation, you are out of circle. For squares, move along edges. First of all check if they are straight lines or not (check if pixels are having either same x or y co-ordinates). After that look for a 90 degree change in angle(if you were moving along a horizontal line then at corner y co-ordinate will stop changing and x will start changing).”

With recent advancements in object detection we have moved billion miles ahead of rudimentary technique so described. In present techniques, manual feature extraction is almost neglected, only raw data is passes to powerful algorithm and algorithm take care of automatically extracting features for us.

You can detect objects using a variety of methods, including:

1. **Feature-based object detection**

	Random sample consensus, or RANSAC. RANSAC works by estimating mathematical model of data contains outlier. Then it ignores outlier and predict using rest of the data.
	RANSAC is accomplished with the following steps : 

	1. Randomly selecting a subset or subspace of the data set
	2. Fitting a predefined model to the selected subset
	3. Determining the number of outliers
	4. Repeating steps 1-3 for a prescribed number of iterations

	RANSAC is primarily used in generating stereo vision.

2. **Viola-Jones object detection**

	This method is popularly used in face detection. All human faces share similar property, these regularity can be matched using Haar features. For more information regarding these features, refer Wikipedia.

3. **Image segmentation and blob analysis**

	Image segmentation is the process of dividing an image into multiple parts. This is usually used to identify objects or other relevant information from images. Segmentation can be performed using any of the relevant algorithm from below given list 1. Otsu’s method 2. K-means clustering 3. watershed segmentation 4. texture filters

	With invent of Convolution Neural Network and GPU computing, many things became past. CNN is proved to be of Ace in area of image processing. CNN eliminated roughly 80% of manual feature selection and mathematical tuning.

	Visual Geometry group form Department of Engineering Science, University of Oxford applied CNN on [ImageNet](http://www.image-net.org/) dataset and achieved state of are [results](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Image Net is a huge dataset with 14,197,122 labelled images. As a part of competition [ILSVRC-2014](http://www.image-net.org/challenges/LSVRC/2014/), provided images of ImageNet subset had to be classified in to 1000 classes. A convolution network based architecture known as VGG16 achieved highest accuracy in the above said compaction.

	Training VGG16 network form ImageNet dataset would require state of art GPUs, unfortunately I cannot afford those, so I found a shortcut. Whenever we train a model, model memorize it in the form of weight in connecting layers. Fortunately, Visual Geometry Group at oxford had provided trained weights in form of model on their official site.

So our task to identify object goes in this way.

1. Get Keras compatible model of VGG16.
2. Construct VGG16 network in Keras
3. Arrange all weights in network so constructed
4. Take a random image
5. Resize and reshape the image as per required dimension by model
6. Get objects in image identified
7. Print name of top 5 object on image with their percentage of confidence as provided by network - optional step

It’s very clear from the discussion that we are not going to train a model, we will be just predicting using someone’s trained model. Let’s start step by step.

1. **Get Keras compatible model of VGG16.**

	I got the model form some known source and kept it in my personal drive. You may obtain the model form my google drive.

2. **Construct VGG16 network in Keras**

   In Keras VGG16 can be defined as follow.
	
	```python
	model = Sequential()
	model.add(ZeroPadding2D((1, 1), batch\_input\_shape=(1,3, 224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))
	```

   This architecture can be visually represented by image given below:

	<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png/v1/fill/w_658,h_386,al_c,lg_1/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png"></p>

	<p align="center">Figure 1. VGG16 model architecture</p>

3. **Arrange all weights in network so constrructed**

	weights are present in *.h5 file, which is standard model file compatible to Keras. Before putting weights to the model, we need to compile the model. Once weight are places at proper place we are good to go with prediction utilizing this model.

	```python
	# defining sgd
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# compiling model
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	# loading pre-trained weights into model
	model.load\_weights('model/vgg16\_weights.h5')
	```

4. **Take a random image**

	I have taken few random images, we will see those images and prediction made on those images shortly.

5. **Resize and reshape the image as per required dimension by model**

	I have taken images of any shape (random images from image search). To input in to CNN, we need to resize it to require dimension i.e. 224*224 pixel.

	A simple snippet of code does this. Image is reshaped in to as (1,3,224,224). where 1 represents one image. 3 represents RGB channel of image.

	```python
	img = image.load\_img(img\_path, target_size=(224, 224))
	x = image.img\_to\_array(img)
	x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
	\# print "SHAPE OF THE IMAGE", x.shape
	```

6. **Get objects in the image identified**

	This step is not a big job to do. Just provide your resized and reshaped image to model and model will provide you with probability of all 1000 classes for given image. It is not necessary that image should have 1000 objects in it. Object which are present in image, model will assign higher probability to these objects compared to others. Out task is to find out such top 5 classes with highest probability.

	```python
	# making a dictionary of class and corresponding probability
	tempDict = {}
	count = 0
	for eachprob in probabilities:
	    tempDict[eachprob] = count
	    count = count +1
	top = 0
	# sorting to get top 5 predictions
	topPredictions={}
	for eachkey in sorted(tempDict,reverse=True):
	    Name = class_names[tempDict[eachkey]]
	    Percentage = eachkey
	    topPredictions[Name]=round(Percentage*100,3)
	    if top == 5:
	        break
	    top = top +1
	```

7. **Last and optional step, Print name of top 5 object on image with their percentage of confidence as provided by network.**

	This can be accomplished by using Python Image Library (PIL). PIL requires fonts, those which are to be used to write on image, I have provided the same on GitHub along with rest of the code.

	```python
	path, filename = os.path.split(imagePath)
	img = Image.open(imagePath)
	draw = ImageDraw.Draw(img)
	fontHeight = 24
	font = ImageFont.truetype("fonts/adventpro-bold.ttf", fontHeight)
	verticleDistance = 0
	predictedClassSorted = {v: k for k, v in predictedClass.iteritems()}
	# Writing on image 
	for eachkey in sorted(predictedClassSorted,reverse=True):
	     # print eachkey, predictedClassSorted[eachkey]
	     draw.text((0, verticleDistance), str(eachkey)+" %" + " : "+str(predictedClassSorted[eachkey]) , (0, 0, 0), font=font)
	     # maintaining verticle distance betweeen two consecutive prediction while writting on image
	     verticleDistance = verticleDistance+fontHeight
	img.save('processedImages/'+filename)
	```

**Conclusion**

I have tried to predict on some of the images. We will see each image one by one.

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_491a14aa5e8e49379373af565d9c55df~mv2.jpg/v1/fill/w_749,h_497,al_c,lg_1,q_85/884a24_491a14aa5e8e49379373af565d9c55df~mv2.webp"></p>

<p align="center">Figure 2. Object Detection with Convolution Neural Network. (Prediction are correct)</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_1ad1b2c235f84cf6bd484805794c8b73~mv2.jpg/v1/fill/w_565,h_662,al_c,q_85/884a24_1ad1b2c235f84cf6bd484805794c8b73~mv2.webp"></p>

<p align="center">Figure 3. Object Detection with Convolution Neural Network. (Prediction is correct, this flower is known as yellow lady slipper)</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_38fc8454707d47559b90a30aeba889e1~mv2.jpg/v1/fill/w_600,h_559,al_c,lg_1,q_80/884a24_38fc8454707d47559b90a30aeba889e1~mv2.webp"></p>

<p align="center">Figure 4. Object Detection with Convolution Neural Network. (well this picture was little confusing, still it is predicted correctly. although it looks like ice but very much simmilar to sea shore)</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_85e238cea9434fa79db708d50cf81006~mv2.jpg/v1/fill/w_900,h_545,al_c,lg_1,q_85/884a24_85e238cea9434fa79db708d50cf81006~mv2.webp"></p>

<p align="center">Figure 5. Object Detection with Convolution Neural Network. [very precisely identified as half track jeep]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_d12afc3331a245f8831dfc953edfcee0~mv2.jpg/v1/fill/w_640,h_640,al_c,q_85/884a24_d12afc3331a245f8831dfc953edfcee0~mv2.webp"></p>

<p align="center">Figure 6. Object Detection with Convolution Neural Network. [Relatively correct prediction, although it's a phone but very much similar to mentioned electronics devices.]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_f2811c97f30c4715a57e37541b7d4343~mv2_d_1827_1500_s_2.jpg/v1/fill/w_945,h_776,al_c,q_85,usm_0.66_1.00_0.01/884a24_f2811c97f30c4715a57e37541b7d4343~mv2_d_1827_1500_s_2.webp"></p>

<p align="center">Figure 7. Object Detection with Convolution Neural Network. [First prediction is "triceratop", which is an animal with horn on head. this object is similar to it. Piggy bank is also correct]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_79812a599f754af2bcfb3d6f0e8aaa8d~mv2.jpg/v1/fill/w_385,h_256,al_c,lg_1,q_80/884a24_79812a599f754af2bcfb3d6f0e8aaa8d~mv2.webp"></p>

<p align="center">Figure 8. Object Detection with Convolution Neural Network. [Well this was a confusing image and it seems was predicted correctly, although it is a bottle but the way it is kept looks like teapot.]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_2632638067c64cf2abfdd66b9edd2993~mv2.jpg/v1/fill/w_696,h_449,al_c,lg_1,q_80/884a24_2632638067c64cf2abfdd66b9edd2993~mv2.webp"></p>

<p align="center">Figure 9. Object Detection with Convolution Neural Network. [Predicted correctly. although it is white crow but still predictions are very much relevant.]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_6bce6e9a17104ba28cb0ec5e484c746d~mv2.jpg/v1/fill/w_945,h_630,al_c,q_85,usm_0.66_1.00_0.01/884a24_6bce6e9a17104ba28cb0ec5e484c746d~mv2.webp"></p>

<p align="center">Figure 10. Object Detection with Convolution Neural Network [Tougher image with lot of interference . old truck was detected correctly but could not detect human.]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_90e6573fe5b049fd988f0818ea90c209~mv2.jpg/v1/fill/w_945,h_623,al_c,q_85,usm_0.66_1.00_0.01/884a24_90e6573fe5b049fd988f0818ea90c209~mv2.webp"></p>

<p align="center">Figure 11. Object Detection with Convolution Neural Network [Predicted correctly]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_ce8c7227c6424590912f69577b5af902~mv2.jpg/v1/fill/w_363,h_272,al_c,lg_1,q_80/884a24_ce8c7227c6424590912f69577b5af902~mv2.webp"></p>

**Figure 12. Object Detection with Convolution Neural Network [Little tough image, ambiguous prediction]
**
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_58691bd084e84c56847ce807bdb2994d~mv2.jpg/v1/fill/w_416,h_238,al_c,lg_1,q_80/884a24_58691bd084e84c56847ce807bdb2994d~mv2.webp"></p>

<p align="center">Figure 13. Object Detection with Convolution Neural Network [wrong]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_fb59519d5dd64449a606150b8f40d372~mv2.jpg/v1/fill/w_319,h_309,al_c,lg_1,q_80/884a24_fb59519d5dd64449a606150b8f40d372~mv2.webp"></p>

**Figure 14. Object Detection with Convolution Neural Network [wrong]
**
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_e70c10e0471949499e4d4c17362224ae~mv2.jpg/v1/fill/w_703,h_454,al_c,lg_1,q_80/884a24_e70c10e0471949499e4d4c17362224ae~mv2.webp"></p>

<p align="center">Figure 15. Object Detection with Convolution Neural Network [wrong]</p>

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_5bdb089c7bf04847b2c2739ebb756431~mv2.jpg/v1/fill/w_768,h_412,al_c,lg_1,q_80/884a24_5bdb089c7bf04847b2c2739ebb756431~mv2.webp"></p>

<p align="center">Figure 16. Object Detection with Convolution Neural Network [wrong]</p>

It is quite evident that, although we have taken great leap in image processing with CNN. We are still not able to identify multiple objects clearly in the same picture. Image quality obviously matter most, small images with clear objects are not labelled properly.
