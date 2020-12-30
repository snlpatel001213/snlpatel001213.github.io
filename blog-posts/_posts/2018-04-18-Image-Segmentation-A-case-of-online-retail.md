---
layout: post
title: "Image Segmentation - A case of online retail"
img: image_segmentation_1.webp # Add image post (optional)
date: 2017-07-03 12:55:00 +0300
description: Understanding Image Segmentation. Image Segmentation done using VGG-CAM technique. # Add post description (optional)
tag: [Image Segmentation, Python]
comments: true
share: true
---

>All codes related to this blog can be found on my [GitHub](https://github.com/snlpatel001213/algorithmia/tree/master/convnet/vggcam) page
Before running code, download following dependancies and place in the same working directory
>1. Download VGG_ 16 model weights from [Here](https://drive.google.com/file/d/0B5mEsS-c9HHAdUNtWmo2NDltaTA/view?usp=sharing)
>2. Download required images from [Here](https://1drv.ms/u/s!Atn7BMbmwAZ4h4NG7kcqmTiNC4ExfQ) (unzip and place the folder in current working directory)
>3. Download annotation files from [Here](https://1drv.ms/u/s!Atn7BMbmwAZ4h4NEwv5o291y0k99ww) (unzip and place the folder in current working directory)
>4. In case of any error, first check requirements.txt file. check wheather any updated package is giving error? (mostly Keras)

In this tutorial, we will practically see, how to identify an object in the image but also to locate it where it exists in the image. Before I tell you anything I want you to look at below given image whereby entire concept in pictured. The same thing we want to achieve- technically called as Image Segmentation/Image localization.

<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_036f7f0c135c451691d576a9548a7f21~mv2.png/v1/fill/w_945,h_355,al_c,lg_1/884a24_036f7f0c135c451691d576a9548a7f21~mv2.png">
</p>

__Figure 1. Image segmentation thereby separating different object class wise in picture.__

Techniques for image segmentation/object localization have come a long way. A decade earlier we used to have techniques like Haar, Shift, and Hog for Image segmentation. With invent and advancement of GPU based CNN architecture a revolution in images segmentation started and soon giants like Facebook and Microsoft jumped into it. Since past 10 years it ws going slow paced but it was accelerated with series of research paper published after 2013 one of them was a paper from MIT -“[Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)” by Bolei Zhou and coworkers in 2014. Soon after this paper Ross Girshick came up with his research paper “[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)” simply known as Regions with Convolutional Neural Network or R-CNN. Soon after this a team at Microsoft came up with more precise and faster algorithm of R-CNN and named it as [Fast-RCNN](https://arxiv.org/abs/1504.08083). Soon after this research, Facebook came up with more precise [Mask R-CNN](https://arxiv.org/abs/1703.06870) and few researchers came up with [Faster R-CNN](https://arxiv.org/abs/1506.01497). Chase to achieve state of art Segmentation still continue but we have achieved a lot in this segment in past 4years.

In this blog I have started with something basic, basically, I am going to implement technique described in a paper a paper “[Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)” by Bolei Zhou and coworkers. This will be a starting point for our next set of implementations on Image segmentation.
<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_6971eaa576cd44abaa9cfaf4c3ab1694~mv2.png/v1/fill/w_945,h_555,al_c,usm_0.66_1.00_0.01/884a24_6971eaa576cd44abaa9cfaf4c3ab1694~mv2.png">
</p>

__Figure 2. Approaches and research in field of image segmentation__

In this tutorial, we will see, How we can implement and utilize image segmentation for online retails. I will be taking example of online retail to understand andimplement this problem step by step. To understand the problem first we need to understand the logic behind How big retail works. Let's take Amazon as an example, this guy is huge with millions of items up and available. To put anything on amazon, may have following procedure

1) A retailer or bulk producer reach out to Amazon.
2) Ask for Amazon for selling something.
3) Amazon grant permissions and access permissions to its sale management console.
4) If that retailer want to sell a “T-shirt for Male” to Amazon he performed following steps In given online form by Amazon, retailers enter relevant details including :
- Valid name
- Product Image
- Its hierarchy in product catalogue, such as a "slim-fit T-shirt" may require following hierarchy to be selected Garments > male > t-shirt > slim fit
- Other information

Now the third point is tricky if you see the millions of products and practically thousand of such category can exists. Now got me ?? Its quite tiresome job and time-consuming too. Considering this. If any a retailer wants to sell 100 products, he needs to follow this step 100 times., I have an idea, If that t-shirt was automatically classified into hierarchy upon uploading its image to Amazon portal, nice isn't it??. In addition to this other related products are identified from photograph and added and properly classified automatically . We could at least save some time and keep out retailer happy and productive too.

Training a machine for all products in the world is a computationally expensive task. This tutorial is really complicated than earlier ones. This involves two massive models and lot of information processing to get the desired output. All codes are available on my GitHub page and I will be explaining each block of code step by step here in this blog.

The over all flow of the blog is having following components :

1) Batch wise Image generator – It's obvious that all Images will not be present in RAM at a time. The job of the batch generator is to load the batch of required Images to RAM and keep it in Numpy format.

2) VGG Network Model,
Here I am defining famous VGG network which was used by Visual Geometry Group, an academic group focused on [computer vision](https://en.wikipedia.org/wiki/Computer_vision "Computer vision") at [Oxford University](https://en.wikipedia.org/wiki/Oxford_University "Oxford University") and  won [ILSVRC-2014](http://www.image-net.org/challenges/LSVRC/2014/) compitition. VGG was trained on 1000 different classes and fine tuning such network on a small specialized dataset (our retail data set) would yield great results.

3) VGGCAM Model
CAM stands for Class Activation Mapping. This model is special and has specialized layer. In this model, all fully connected layer from the bottom of the VGG model is replaced by a convolutional layer and max pooling with the massive size of $14 * 14$. usually we use pool size of $2 * 2$ or $3 * 3$. but when pulled with such a high pool size the weights heat map actually represents the portion of the image which was actually responsible for the prediction of the class. The pulled filter will be multiplied by factor $16$ so that $[14 * 14]$ scale out by multiplying with factor $16$ yields heat map of size $[224 * 224]$ (size of the actual image). Such heat map shows higher activation at the portion of the image which is responsible for prediction of given class of in the image. A class activation map for a particular category indicates the discriminative image regions used by the CNN to identify that category.
The overall architecture of the VGG model Class Activation Mapping model can be shown by the following image.
<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_d33b406f28164dcb8199f0454d5103d0~mv2.jpg/v1/fill/w_945,h_444,al_c,lg_1,q_85/884a24_d33b406f28164dcb8199f0454d5103d0~mv2.webp">
</p>

 __Figure 3. Simplified Image showing how Cass activation mapping works by creators of the “Learning Deep Features for Discriminative Localization”__

4) A class activation map generator
A function which takes a particular class and generates 2D heat map for that class taking weight of last convolution layer of VGGCAM<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_766543488dbd414594e3a542b1ebba18~mv2.jpg/v1/fill/w_752,h_584,al_c,lg_1,q_85/884a24_766543488dbd414594e3a542b1ebba18~mv2.webp">
</p>

__Figure 4. This flow chart explains all procedure for object localization. 1) A batch generator provides images and labels repetitively. 2) A VGG model with pre-trained wights, get fine tuned on provided images with labels 3) at each iteration, from weights of VGG, last fully connected layers are removed and convolutional layer with pull size $14*14$ introduced. 4) Vgg network will predict the class of the test image pass its weights and predicted class to VGGCAM model. VGGCAM model will produce class activation map and class activation generator plot such weights into the 2D heat map.__

Having captured entire idea behind working, let's move to the implementation part.

**1) VGG NETWORK**
```python
def getModelDefination(trainedModelPath=None):
    """
    core definition of model
    :return: compiled model
    """
    # defining convolutional network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
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
    model.add(Dense(2, activation='softmax'))
    # compiling model
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    # returning Model
    return model
```

**2) VGG CAM model**
```python
def VGGCAM(nb_classes, num_input_channels):
    """
    Build Convolution Neural Network
    nb_classes : nb_classes (int) number of classes
    num_input_channels : number of channel to be kept in last convolutional model of VGGCAM
    returns : Neural Net model
    """
    VGGCAM = Sequential()
    VGGCAM.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    VGGCAM.add(Convolution2D(64, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(64, 3, 3, activation='relu'))
    VGGCAM.add(MaxPooling2D((2, 2), strides=(2, 2)))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(128, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(128, 3, 3, activation='relu'))
    VGGCAM.add(MaxPooling2D((2, 2), strides=(2, 2)))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(256, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(256, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(256, 3, 3, activation='relu'))
    VGGCAM.add(MaxPooling2D((2, 2), strides=(2, 2)))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(512, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(512, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(512, 3, 3, activation='relu'))
    VGGCAM.add(MaxPooling2D((2, 2), strides=(2, 2)))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(512, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(512, 3, 3, activation='relu'))
    VGGCAM.add(ZeroPadding2D((1, 1)))
    VGGCAM.add(Convolution2D(512, 3, 3, activation='relu'))
    # Add another conv layer with ReLU + GAP
    VGGCAM.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same"))
    VGGCAM.add(AveragePooling2D((14, 14)))
    VGGCAM.add(Flatten())
    # Add the W layer
    VGGCAM.add(Dense(nb_classes, activation='softmax'))
    # VGGCAM.summary()
    return VGGCAM
```
Note that last fully connected layers of the VGG are replaced by large pooling layer VGGCAM.add(AveragePooling2D((14, 14))).

**3) Fine tuning VGG Model with specialized train set**
As explained earlier, at each iteration a new set of images fine tune VGG model and after training, weights are passed to VGGCAM model, where with the help of large pulling a class activation is developed for the given class.
```python
def train_VGGCAM(trained_weight_path, nb_classes,epoches,batchSize, num_input_channels):
    """
    Train VGG model
    args: VGG_weight_path (str) path to keras vgg16 weights
    nb_classes (int) number of classes
    num_input_channels (int) number of conv filters to add
    in before the GAP layer
    """
    # Load model
    trainedModel = getModelDefination(trainedModelPath=trained_weight_path)
    # Compile
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    trainedModel.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    for epochNo in range(0,epoches):
    print "Epoch No : ", epochNo
    batch_Count = 0
    for image,labels in getImageAndCategory(batchSize):
        try:
            # last 10 image selection for test while training
            # train model with rest images
            for i in range (len(trainedModel.layers)):
                print (i, trainedModel.layers[i].name),
                print "\n"+"%"*100
                trainedModel.fit(image,labels,batch_size=50,nb_epoch=1, verbose=1)
                modelCAM = VGGCAM(nb_classes,num_input_channels)
                print ("NAME OF LAYERS IN NEW MODEL FOR CAM")
                for i in range (len(modelCAM.layers)):
                    print (i, modelCAM.layers[i].name),
                    # Load weights to new model
                    for k in range(len(trainedModel.layers)):
                        weights = trainedModel.layers[k].get_weights()
                        modelCAM.layers[k].set_weights(weights)
                        # modelCAM.layers[k].trainable=True
                        if k==16:
                            break
                            print('\nModel loaded.')
                            batch_Count = batch_Count + 1
                            modelCAM.save_weights("CAM_Trained.h5")
                            # to see performance of model on one of the image while training
                            plot_classmap("CAM_Trained.h5",trainedModel, "jeans.jpg", 1,nb_classes,num_input_channels)
                        except:
                            print traceback.print_exc()
```

**4) Getting Heat MAP**
``` python
def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):
    """
    To get heat map from the weight present in last convolutional layer in VGGCAM network
    """
    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
    batch_size=batch_size,
    num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized, (1, -1, 224 * 224))
    classmap = K.dot(WT, conv6_resized)
    # print "\n"+"$"*50
    classmap = classmap.reshape((1, nb_classes, 224, 224))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])
```

**5) Potting heat map**
```python
def plot_classmap(VGGCAM_weight_path, trainedModel,img_path, label,nb_classes, num_input_channels, ratio=16):
    """
    Plot class activation map of trained VGGCAM model
    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
    img_path (str) path to the image for which we get the activation map
    label (int) label (0 to nb_classes-1) of the class activation map to plot
    nb_classes (int) number of classes
    num_input_channels (int) number of conv filters to add
    in before the GAP layer
    ratio (int) upsampling ratio (16 * 14 = 224)
    """
    # Load and compile model
    modelCAM = VGGCAM(nb_classes, num_input_channels)
    modelCAM.load_weights(VGGCAM_weight_path)
    modelCAM.compile(loss="categorical_crossentropy", optimizer="sgd")
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    #vgg model is used to predict class
    label = trainedModel.predict_classes(x.reshape(1, 3, 224, 224),verbose=0)
    batch_size = 1
    classmap = get_classmap(modelCAM,x.reshape(1, 3, 224, 224),nb_classes,batch_size,num_input_channels=num_input_channels,ratio=ratio)
    classes = ["jeans","tshirt"]
    print "PREDICTED LABEL : ", classes[label[0]]
    plt.imshow(img)
    activation = classmap[0,0, :, :]+classmap[0,1, :, :]
    plt.imshow(activation,
    cmap='jet',
    alpha=0.5,
    interpolation='nearest')
    plt.show()

```
When I progressively checked performance improvement in the model epoch by epoch, I got following result, It is very clear that if such large amount of specific examples are provided it can actually perform great.

<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_700e66406e174b4bb337c04db223cf11~mv2.jpg/v1/fill/w_728,h_728,al_c,q_85,usm_0.66_1.00_0.01/884a24_700e66406e174b4bb337c04db223cf11~mv2.webp">
</p>

__Figure 5. Progressive object localization with VGGCAM to localize T-shirt in the picture (Red flare represents region being responsible for prediction of class T-shirt)__

I do understand, that this requires a lot of training, I have tried to develop a prototype. To this prototype, one can provide N number images and M number of classes to make the prediction. To train such model it would require highly powerful, state of art GPUs.
