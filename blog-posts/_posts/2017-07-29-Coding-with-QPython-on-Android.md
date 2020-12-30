---
layout: post
title: "Coding with QPython on Android Devices"
img: qpython.webp # Add image post (optional)
date: 2017-07-29 12:55:00 +0300
description: Understanding Image Segmentation. Image Segmentation done using VGG-CAM technique. # Add post description (optional)
tag: [Image Segmentation, Python]
comments: true
share: true
---

I am a programming enthusiast. One or another program always keep running on my PC anytime any day.
Last time I was working with a  web scrapper. Everything was good but some parameters were out of control.

1. Electricity cut, being in India this is the most common hurdle. Laptop battery can last up to 2.5 hr but WiFi doesn't work without electricity.  
2. While I remain at the office and your personal work is in progress at home, it is very difficult to keep eye on. Sometimes I realize that just after I left for office the program encountered an error or other failure and my entire day gone wasted.

To make peace with all these issues, I found a new way with “QPython”. "QPython is a script engine which runs Python programs on Android devices. It also can help developers develop Android applications.”  
In short Q Python is a Python compiler that can run your Python scripts on any Android device. This informative blog would be very short and we will move forward step by step till you get an exact idea on how to run your python scripts on the android device even while you are enjoying road trip or paragliding. 

# Installation #

Installing Q Python is as easy as you install any other application from play store. Go to plays store, search for  Q Python and here it is.

![](https://static.wixstatic.com/media/884a24_d1ffa99a46b74990af81a76e3948629d~mv2.jpg/v1/fill/w_567,h_449,al_c,q_80,usm_0.66_1.00_0.01/884a24_d1ffa99a46b74990af81a76e3948629d~mv2.webp)

    Figure 1. QPython in Android play Store.

# Working with QPython #

Console, where you can code as you do in normal python console.  I found this functionality of little use as it is very inconvenient to type longer code on phone’s keyboard. The console can be used effectively to test small-small snippet of code. 

![](https://static.wixstatic.com/media/884a24_dfa67bdc86764819a9367113c06ec3ef~mv2.jpg/v1/fill/w_567,h_1008,al_c,q_85,usm_0.66_1.00_0.01/884a24_dfa67bdc86764819a9367113c06ec3ef~mv2.webp)

Figure 2.  QPython native console

Editor window is the second functionality. This is almost full flagged python editor. It supports auto indentation,  code highlight, open, save, edit, run, text wrapping, search and goto line like basic functions. Editor functionality is not limited to just simple scripts but it does allow to manage the entire project where code may span to multiple files. You can manage entire projects as we do with Pycharm or Spyder or any other IDE.

![](https://static.wixstatic.com/media/884a24_8c15a87d73414b13ae801651a048f152~mv2.jpg/v1/fill/w_567,h_1008,al_c,q_85,usm_0.66_1.00_0.01/884a24_8c15a87d73414b13ae801651a048f152~mv2.webp)

Figure 3.  QPython native Editor

Libraries, allow us to manage packages. Installing packages is supported with both pip \[python official package installer\] and custom QPython package manager.

![](https://static.wixstatic.com/media/884a24_80472433c32e49b68d52050d2fcb5724~mv2.jpg/v1/fill/w_567,h_1008,al_c,q_85,usm_0.66_1.00_0.01/884a24_80472433c32e49b68d52050d2fcb5724~mv2.webp)

Figure 4.  QPython native Libraries manager.

What I saw particularly while installing packages to QPython is

1.  Small or large packages which are written in Python get installed without any hurdle. Example requests, beautifulsoup, wikipedia, bottle etc.
    
2.  Package with dependencies like extra model files and have core functional written in Cython or C sometimes fails to get installed. Example Numpy, Tensorflow etc

**Usages / Productivity**

1. QPytohn can be very useful for writing small snippets for the following purpose.

2. Web scrappers which takes a long time and due to uneven data often require to monitor

3. 'bottle' package you can make your phone a web server.

4. Many native Android functions can be controlled through QPython such as:

    1. Alarm me when battery charged to 95%  or drained to 5%.

    2. Alarm me/toast a notification when web crawler program is halted due to error.

    3. Alarm me/toast a notification when I am out of Network and crawler is paused for a while. Crawler again continues when the network is detected.

Practically I have used QPython to crawl millions of pages from Wikipedia and duck duck go. I am exploring more possibilities, will continue to extend this blog as an when I will explore more things.