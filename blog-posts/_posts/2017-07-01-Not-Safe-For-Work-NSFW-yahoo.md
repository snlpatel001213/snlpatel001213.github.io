---
layout: post
title: "Yahoo's Not Safe For Work (NSFW)"
img: not-safe-for-work.webp # Add image post (optional)
date: 2017-07-9 12:54:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
tag: [keras, GPU, Python, Machine Learning Advance]
comments: true
share: true
---

----
1. To run this tutorial you will surely require a GPU, you may use AWS P2 GPU instances use NVIDIA Tesla K80 GPUs. For to "[How to set up AWS  infrastructure for machine Learning](https://www.machinelearningpython.org/single-post/AWS-GPU-for-machine-learning)"    
2. Code discussed here can be found on my [GitHub](https://github.com/snlpatel001213/algorithmia/tree/master/convnet/NSFW) repository.
3. All safe and unsafe images can be found [here](https://drive.google.com/file/d/0B5mEsS-c9HHAdThlZ1JFbDFtNUE/view?usp=sharing) [for educational purpose only] (password - nsfw)
4. Final model generated after training can be found [here](https://drive.google.com/file/d/0B5mEsS-c9HHAcHliMjRNdWV2aTg/view?usp=sharing). You may use this model for further training and fine tuning on your custom data-set.
5. If you get any error then, please check with [requirements.txt](https://github.com/snlpatel001213/algorithmia/blob/master/convnet/NSFW/requirements.txt) file to check your python package versions.
6. Code Compatibility : python 2.7 , tested on ubuntu 16.04 with theano as backend
----

Not Safe For Work (NSFW) is a class of content that is unstable for minor or at public places. NSFW mainly involves porn/Adult content. Presently NSFW content is majorly filtered through source based approaches. Source based approaches means blocking appropriate URL/ feed which is known source of such content. International Foundation for Online Responsibility (IFFOR) on 15 April 2011 initiated a domain called .xxx for adult content, so that on the basis of users preference such content can be blocked. wider acceptance of such domain would take time. So it better we apply science to this problem.
Present approaches to detect nudity depends on following:
1. URL Name based blocking.
2. File Name based blocking.
3. Page Name based blocking.
4. keyword based blocking.

However these approaches fails many time due to following reasons:
1. Ambiguous URLs.
2. Ambiguous File Name/ Page Name.
3. With Ever changing nature of Internet and billions of digital device pushing data continuously, Its becomes even difficult to track such content

On SEP 30th 2016 [Yahoo](https://github.com/yahoo/open_nsfw) open sourced a model that is capable of differentiating NSFW content form SAFE FOR WORK (SFW)  content. In this blog post I will walk you through step by step procedure to train a network to differentiate NSFW and SFW content.
The tutorial is divided  in to following sub-parts for easy understanding:

1. Data Collection
2. Peeping inside collected data
3. Data Preprocessing
4. Data set size and Distribution
5. Examining results

# Data Collection #

Yahoo doesn't provide data-set which was used for their research work, So I  made one of mine. Here is how I did it.

1. NSFW images collection (+ve):
    You may go to any site contain adult content and scrap it.  SIMPLE!! Isn't it ?? I will neither be describing the actual proceeds nor provide any program for the same because of known copyright / abuse troubles. 
2. SFW images collection (-ve) : 
    In fact this part is very simple. in Mozilla Firefox download a plugin named as "Flashgot". Now go-to any popular image search site like tumbler, google images, bing Image search and search with keywords like "couple", "public kissing", "love couple". Download all images from given page using Flashgot.
    To download  NSFW and SFW images which I have used for present tutorial click here [password : nsfw]
    Note that provided images are for Education purpose only, It doesn't carry any monetary intention. 

# Peeping Inside Collected Data #

It is very essential to check quality of the data to ascertain the quality of resulting model. Images for NSFW is very clear with very less noise as I have collected these from Adult site. Images for  SFW is considerably noisy. As SFW is collected from search engines and sometime images not at all related to our purpose are also captured. Any ways we got to move ahead with this data we cannot get clean data without manual curation. Impurity in data will lower down our accuracy but this exercise will surely provide an intuition that our approach really works. [You may go further on cleaner data, on your own]
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_d5e6fa1aec5043dc88af7fadca20260a~mv2.jpg/v1/fill/w_567,h_567,al_c,q_80,usm_0.66_1.00_0.01/884a24_d5e6fa1aec5043dc88af7fadca20260a~mv2.webp"></p>

 <p align="center">Figure 1. Totally unrelated images (junk) to our purpose collected in SFW Images</p>
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_3dbd364e60c34be0acc5c98f7f77365a~mv2.jpg/v1/fill/w_567,h_567,al_c,q_80,usm_0.66_1.00_0.01/884a24_3dbd364e60c34be0acc5c98f7f77365a~mv2.webp"></p>

 <p align="center">Figure 2. NSFW image collected in SFW Images</p>

Figure 2. and Figure 3. shows that our SFW Image data-set is not clean, but lets move ahead with this data only.

# Data Pre-processing #

Download above said data-set and put in  working directory before going ahead with coding
1. Importing Requirements \- while experimenting with things I have imported many unwanted packages, You may remove them. 
    ```python
    import os.path as path  
    import matplotlib.pyplot as plt  
    import numpy as np  
    import theano  
    import keras  
    import traceback  
    from keras.models import Sequential  
    from scipy.misc import imread, imresize, imsave  
    from keras.layers.core import Flatten, Dense, Dropout,Activation  
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, UpSampling1D, Cropping2D  
    from keras.optimizers import SGD  
    from keras.utils.np_utils import to_categorical  
    %matplotlib inline  
    from keras import backend as K  
    from keras.utils import np_utils  
    K.set\_image\_dim_ordering('th')  
    import traceback  
    from scipy import ndimage  
    from sklearn.cross_validation import train\_test\_split
    ```
2. Resizing Images \- Collected images are of various size and  extension, In this step we will convert images to 224*224 dimension.After invoking below given functions, each image will be resized and placed in SAFE\_resized and UNSAFE\_resized folder. 
    ```python
    def imageResize(basename,imageName):  
        """  
        resize image  
        basename : eg. /home/sunil/fishes/bet  
        imagename : xyz.jpg  
        """  
        new_width  = 224  
        new_height = 224  
        try:    
            img = Image.open(basename+"/"+imageName) # image extension *.png,*.jpg  
            img = img.resize((new\_width, new\_height), Image.ANTIALIAS)  
            img.save(basename+'_resized/'+imageName)  
        except:  
            os.mkdir(basename+'_resized/')  
            img = Image.open(basename+"/"+imageName) # image extension *.png,*.jpg  
            img = img.resize((new\_width, new\_height), Image.ANTIALIAS)  
            img.save(basename+'_resized/'+imageName) 
    def resizer(folderPath):  
        """  
        to resize all files present in a folder  
        resizer('/home/sunil/imageTagging/data/allCats_resized/')  
        resizer('/home/sunil/imageTagging/data/allCats_resized/')  
        """   
        for subdir, dirs, files in os.walk(folderPath):  
            for fileName in files:  
                try:  
            #         print os.path.join(subdir, file)  
                    filepath = subdir + os.sep + fileName  
                    if filepath.endswith(".jpg"):  
                        imageResize(subdir,fileName)  
                except:  
                    print traceback.print_exc()  
                    os.remove(subdir+"/"+fileName) 
    \# Actually applying  resizing to images  
    resizer('/home/sunil/imageTagging/downloadData/SAFE')  
    resizer('/home/sunil/imageTagging/downloadData/UNSAFE')
    ```
3. Loading images as Numpy array 
    ```python
    def load_image( infilename ) :  
        """  
        load image from disk  
        :param infilename:  
        :return:  
        """ 
        img = ndimage.imread( infilename )  
        data = np.asarray( img, dtype="int32" )  
        resized = data.reshape(data.shape[2],data.shape[0],data.shape[1])  
        return resized
    def turnToNumpy(folderPath):  
        """  
        turn stored images on disk to numpy  
        turnToNumpy('/home/sunil/imageTagging/  
        """  
        temp = []  
        for subdir, dirs, files in os.walk(folderPath):  
            for fileName in files:  
                    try:  
                        filepath = subdir + os.sep + fileName  
                        if load_image(subdir+"/"+fileName).shape == (3,224,224):  
    \#                         print filepath  
                            temp.append(load_image(subdir+"/"+fileName))  
                    except:  
                        os.remove(subdir+"/"+fileName)   
        return np.asarray(temp) 
    SAFE = turnToNumpy('/home/sunil/imageTagging/downloadData/SAFE_resized')  
    UNSAFE = turnToNumpy('/home/sunil/imageTagging/downloadData/UNSAFE_resized')
    ```
# Data set size and Distribution #

I have  5805 safe images and 8081 unsafe images, total about 11610. Out of this randomly 70 % images (8127) will go to Train and remaining 30% images (3483) will go to test.

# Model Definition #

Again! I have taken a popular VGG-16 network. Stochastically but It does perform well on images.
<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png/v1/fill/w_658,h_386,al_c,lg_1/884a24_c506f4dce6fd4163b84355297f4a2a3a~mv2.png"></p>

<p align="center">Figure 3. VGG16 network architecture.</p>

```python
    \# defining model  
    model = Sequential()  
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))  
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
    mod5el.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(MaxPooling2D((2,2)))  
    model.add(Flatten())  
    model.add(Dense(128, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(64, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(2, activation='softmax'))  
    \# model.summary()
```

Output layer will be giving us any of the  classes, 1 - SFW and 2 - NSFW. I have taken stochastic gradient descent as optimizer function. Categorical cross entropy was choose as loss function as it perform well on multi class classification. 

```python
    \# compiling and fitting model
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X\_train, y\_train, batch\_size=10, nb\_epoch=10, verbose=1, validation\_data=(X\_test, y_test))
```

# Examining results #

   Although accuracy was recorded continuously as Performance matrix, we require  to visually inspect some of the images to actually get idea about performance. accuracy for the entire dataset was found to be 72%. 
   Below given code snippet will save all test data-set images to disk with actual and predicted class label.

```python
    \# writing test images to disk  
    #the name of the file would be imageNumber\_actualClass\_predictedClass.png 
    imageNumber = 0 
    for imageNumber in range (0,len(X_test)): 
        tempX = X\_test[imageNumber].reshape(X\_test[imageNumber].shape[1], X\_test[imageNumber].shape[2], X\_test[imageNumber].shape[0]) 
        # plt.show() # to show image here as well 
        predicted = pr[imageNumber] 
        if (int(actual[imageNumber]) == 1 and int(predicted)== 1): 
            imsave("tp/"+str(imageNumber)+"_"+str(actual[imageNumber])+"_"+str(predicted)+".png",tempX) 
        if (int(actual[imageNumber]) == 0  and int(predicted)== 0): 
            imsave("tn/"+str(imageNumber)+"_"+str(actual[imageNumber])+"_"+str(predicted)+".png",tempX) 
        if (int(actual[imageNumber]) == 0  and int(predicted)== 1): 
            imsave("fp/"+str(imageNumber)+"_"+str(actual[imageNumber])+"_"+str(predicted)+".png",tempX) 
        if (int(actual[imageNumber]) == 1  and int(predicted)== 0): 
            imsave("fn/"+str(imageNumber)+"_"+str(actual[imageNumber])+"_"+str(predicted)+".png",tempX)
    
```

<p align="center"><img class="img-responsive" src="https://static.wixstatic.com/media/884a24_c20a4eab29034ce68e5a42223822dee7~mv2_d_4000_3000_s_4_2.jpg/v1/fill/w_860,h_645,al_c,q_85,usm_0.66_1.00_0.01/884a24_c20a4eab29034ce68e5a42223822dee7~mv2_d_4000_3000_s_4_2.webp"></p>

<p align="center">Figure 4. Actually classified images in as (True Positive, True Negative, False Positive and False Negative)</p>

Above given image summarizes performance of our model on test data-set. Accuracy on test test data-set was found to be ~72%, that implies algorithm is performing well on test data-set. lets examine where we re missing,

1. False Negative - these are sample which were originally positive (unsafe) and classified as negative (safe). Actually when you look at images will find that algorithm did quite a good job.  All images classified as safe are actually safe. As we have scrapped this data without manual curation these safe  images  were by mistake given as unsafe.

2. False Positive - these are sample which were originally positive (safe) and classified as negative (safe). Actually when you look at images will find that algorithm did quite a good job.  All images classified as unsafe are actually unsafe. As we have scrapped this data without manual curation these unsafe  images  were by mistake given as safe.

Scope of improvement:

1. More data , the most required thing. I have conducted experiment with 11000 images and still performs good. Similar experiment if repeated with millions of images it will perform the best.

2. Quality of images - As we have seen, we are loosing on accuracy due to mis-tagged images. If training set with good quality is taken, it will surely improve results.
Here I have used training from beginning, next time I will demonstrate much better model using transfer learning.
