---
layout: post
title: Setting up environment for C and C++
description: This blog covers how to setup environment for C and C++ project.
image:  
  path: /assets/img/blog/trezorglitch.png
canonical_url: http://bit.ly/trezor-glitch
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

# Setting up C/C++ environment for any project

There are many languages but there is no match for the simplicity and speed that any C/C++ program offers.   It not easy to setup the perfect developer environment for as it is with other language like python and R. 

There are many components to any C/C++ environment, it includes : 

1.  An IDE (Intelligent Development Environment) 
2. A build system



## IDE -  Visual Studio Code

There are many IDE like Visual Studio, Code Blocks, Pycharm Clion. I personally like Visual Studio Code because it is lightweight, and it has plethora of features. You can download Visual Studio Code [here](https://code.visualstudio.com/), you will land on a page like this: 

![image-20210101121432972](/home/supatel/.config/Typora/typora-user-images/image-20210101121432972.png)

## Setting up VSC for C++ development

The first time you open VSC you will see a welcome window. VSC has a very simple layout: a bar on the left with 5 buttons (File explorer, Find, Git integration, Debug, Extensions), a status bar on the bottom and a window with tabs for the editors. Click the last button to open Extensions:
[![img](https://res.cloudinary.com/practicaldev/image/fetch/s--2PfYlUyd--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/2_searching_extensions.png)](https://res.cloudinary.com/practicaldev/image/fetch/s--2PfYlUyd--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/2_searching_extensions.png)

To develop C++ we will install two extensions, the first one is C/C++, which is already shown in the last figure, to install it just click the green button that says `Install`:
[![img](https://res.cloudinary.com/practicaldev/image/fetch/s--B2HAFyKB--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/3_cpp_extension_install.png)](https://res.cloudinary.com/practicaldev/image/fetch/s--B2HAFyKB--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/3_cpp_extension_install.png)

Then we will search for "easy c++" and then install the extension called "Easy C++ Projects"
[![img](https://res.cloudinary.com/practicaldev/image/fetch/s--5Ipp_BRO--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/4_easy_cpp_extension_install.png)](https://res.cloudinary.com/practicaldev/image/fetch/s--5Ipp_BRO--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/4_easy_cpp_extension_install.png)

------

## Environment setup finished

After installing the extensions, a blue button will appear saying `Reload`, clicking it will reload the window and activate the extensions we just installed, as shown here:
[![img](https://res.cloudinary.com/practicaldev/image/fetch/s--ohlBVV4x--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/5_extensions_installed.png)](https://res.cloudinary.com/practicaldev/image/fetch/s--ohlBVV4x--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/acharluk/UsefulStuff/master/programming/C%2B%2B/images/1/5_extensions_installed.png)

Good job! Now we have an environment for depeloping our first C++ project!