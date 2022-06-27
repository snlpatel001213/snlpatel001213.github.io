---
layout: page
title: Deep Learning
description: My expertise in Deep Learning
image: assets/images/pic01.jpg
nav-menu: false
description: null
image: null
author: null
show_tile: false
---
<section id="one">
	<div class="inner">
		<header class="major">
			<h1>DeeP Learning</h1>
		</header>
        <h2 id="content">The beginning @ IIITA</h2>
        <div class="row 200%">
            <div class="8u 12u$(medium)">
                <p>I started exploring DL in 2013, pretty early days for deep learning software stack. Back then libraries like Tensorflow, Pytorch, Keras we not avaialble.
                    I started building layers (Forward propagation) and used library for backpropagation(Autograde). Back then the layer were simpler, mosly linear + activation.</p>
                    <p>My first project was to find if tro protein interacts or not. Protein are 3D molecure acts like micro machies and keep us alive and kicking. When protein protein interaction goes wrong it may ends up causing cancer</p>
                    <p>This project of mine could predict protein protein interaction with over 95% Accuracy. To the date, its state of the art.</p>
                    <p>To make my work avaialble to world I started a web server from university. After 7 years this server is still serving and avialble at : <a href="https://bioserver.iiita.ac.in/deepinteract/"> Deep Interact</a> </p>
                    <p> A simillar service predicting iteraction of long non-coding RNA (another component in living beings) was also published and it is avialble as a web service as :   <a href="https://bioserver.iiita.ac.in/deeplnc/index.php"> Deep LNC</a></p>
                <p><b>So these were initial days, learnt a lot and eventually made it a profession</b></p>
            </div>
            <div class="4u 12u$(medium)">
                        <span class="image fit"  style="object-fit: contain" >
                            <a href="{% link assets/pdf/M_Tech_Thesis_maincontent.pdf %}" target="blank_" >
                                <img src="{% link assets/images/thesis.png %}" alt="" />
                            </a>
                        </span>
                        <span>
                            <a href="https://bioserver.iiita.ac.in/deepinteract/" target="blank_" >
                                <img src="{% link assets/images/DeepInteract.png %}" alt="" />
                            </a>
                            
                        </span>
            </div>
        </div>
        <hr>
        <h2 id="content">The first two corporate projects</h2>
        <div class="row 200%">
            <!-- <div class="3u 12u$(medium)">
                    <div class="12u"><span class="image fit"  style="object-fit: contain" ><img src="{% link assets/images/pic08.jpg %}" alt="" /></span></div>
                    <div class="12u"><span class="image fit"  style="object-fit: contain" ><img src="{% link assets/images/pic09.jpg %}" alt="" /></span></div>
                    <div class="12u$"><span class="image fit" style="object-fit: contain" ><img src="{% link assets/images/pic10.jpg %}" alt="" /></span></div>
            </div> -->
            <div class="12u 12u$(medium)">
                <p>As soon as I completed my Master's at IIIT-A, I got in to TCS digital R&D. I just stayed for 8 months but completed 2 projects from conceptualization to development</p>
                <p>1. The first project was to hierarchially categorize given tech query to predefined class,subclass and automatically assign it to resposilbe person.
                This was short of IT automation. There were copuple of challenges with this project 1) Its hierarchial classiifcation 2) There were 3 lavel of classes. The total permutations of getting given query classified turns out to be in order $ 10^5 $. In short.... The success came from hierarchial softmax function</p>
                <p> 2. The second project was very simple but interesting. It was about <b>financial statement reconcilling </b> Same Np hard problem but 1D CNN could figure out patterns in data with 94% accuracy. </p>
            </div>
        </div>
        <hr>
        <h2 id="content">The NLP Journey</h2>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <p>After a stint of deep learning at TCS I worked at two deep tech startup working on NLP. By now, I had alreasdy worked with 1D-CNN and RNN which is very much used in NLP space.</p>
                <p>Here with these startups I worked with task like text Classification, Summarization, Named Entity Resolution, Language translation etc..</p>
                <p>I learnt a lot about different embedding techniques such as Word2Vec, Fasttext, Glove, SkipThrough, Elmo, transformer and Bert </p>
                <p>I also learnt about important concept such as Teacher forcing, Attention layers, Batched inference, Combining CNNs and RNNs, doing training and inference with multiple GPUs, etc</p>
                <p> I started writting my forst book on NLP and Nvidia hired me as Deep Learning Data Scientist</p>
            </div>
            <div class="6u 12u$(medium)">
                    <img src="{% link assets/images/embeddings.png %}" alt="" />
            </div>
        </div>
        <hr>
        <h2 id="content">Life's work @ Nvidia</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
            <p>Joing Nvidia was truelly a life changing experience, there's so much to learn from a company which is literlly a AI/DL powerhouse for all the other companies in the world.</p>
            </div>
            <div class="4u 12u$(medium)">
                <div class="row 100% uniform">
                    <div class="6u"><span class="image fit"><img src="{% link assets/images/amp.png %}" alt="" /></span></div>
                    <div class="6u"><span class="image fit"><img src="{% link assets/images/amp2.png %}" alt="" /></span></div>
                </div>
            </div>
            <div class="8u 12u$(medium)">
                <p>The first asiignment was to do large scale training. Just before I joined Nvidia Volta generation of cards were released, Volta offers mixed precision training. After I completed the first assigment I already knew about Multi-GPU
                    Multi-Node training, and Mixed precision training. Now I can write mixed custom mixed-precision operations for any layer or I can use Automatic Mixed Precision (AMP). AMP almost increases the training speed by 2X.</p>
            </div>
            <!-- row -->
            <div class="8u 12u$(medium)">
                <p>The second project that I was Intelligent Video Analytics (IVA). Precisely the task is to design a scalable GPU-based deep learning inference platform that can run over 1 Million active cameras.
                I acquired skills in model optimization. The inference has its own requirements and the model that is trained should be used in inference as it is. There comes an intermediate step - model optimization.
                I am comfortable with the following operations to optimize a deep learning model :  
                <li>Model profiling to see which layer is compute-heavy. Can we replace this layer with optimized once?</li>
                <li>Is there any major buffer transfer between CPU and GPU, can we reduce that?</li>
                <li>In case of enseble model use pointer based direct buffer transfer within GPU instead of transfering it CPU and GPU back and forth. </li>
                <li>Convert the model to FP16 and check performance as well as the accuracy in the optimized model.</li>
                <li>Use quantization and convert the model to INT8 and analyze performance as well as the accuracy in the optimized model.</li>
            Most of the models that I have optimized till time includes, SSD, YOLO V3, MaskRCNN, YOLO V4 scaled, Unet etc..    
            </p> 
            </div>
            <div class="4u 12u$(medium)">
                <div class="row 100% uniform">
                    <div class="6u"><span class="image fit"><img src="{% link assets/images/deepstream.png %}" alt="" /></span></div>
                    <div class="6u$"><span class="image fit"><img src="{% link assets/images/triton.png %}" alt="" /></span></div>
                    <div class="6u"><span class="image fit"><img src="{% link assets/images/pytorch.png %}" alt="" /></span></div>
                    <div class="6u$"><span class="image fit"><img src="{% link assets/images/tensorrt.png %}" alt="" /></span></div>
                </div>
            </div>
        </div>
        <div class="row 200%">
            <div class="4u 12u$(medium)">
                <div class="row 100% uniform">
                    <div class="6u"><span class="image fit"><img src="{% link assets/images/docker.png %}" alt="" /></span></div>
                    <div class="6u$"><span class="image fit"><img src="{% link assets/images/k8s.jpg %}" alt="" /></span></div>
                    <div class="6u"><span class="image fit"><img src="{% link assets/images/onnx.png %}" alt="" /></span></div>
                    <div class="6u$"><span class="image fit"><img src="{% link assets/images/gdb.png %}" alt="" /></span></div>
                </div>
            </div>
            <div class="8u 12u$(medium)">
                <p> After model optimization theres comes deployment phase, A deployment with video requires additional components such as Stram muxer, Tracker, Demuxer, On screen display, Analytics etc.
                    For IVA pipeline I rely heavily on DeepStream SDK.  I am well experienced at deploying stuff with Deepstream. I have contrributed heavily to DeepStream, few of the features are as follow:
                    <li>Seemless buffer transfer between CPU and GPU in case of ensemble models</li>
                    <li>Tile processing to dectrease compute resource utilization by heavier model</li>
                    <li>A demuxing pipeline to split streams back to original stream after inference</li>
                    <li>Using Primary and secondary inferecne engines</li>
                    <li>Video pipeline profiling</li>
                </p> 
            </div>
        </div>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <p>So far, I have worked and contributed to following Nvidia SDKs..</p>
                <li> Triton - To deploy CUDA, Python, Tensorflow, Pytorch and ONNX models and to create really scalable services</li>
                <li> TensorRT - for model optimization </li>
                <li> TAO - For faster training on GPU </li>
                <li> Deepstream - For building GPU powered IVA pipeline </li> 
                <li> RIVA and NeMo - For building and deploying language module including Text to speech, Automatic Speech recognition and NLP</li>
                <li> Maxine -  For to develop online conferencing application powered by GPU</li>
                <li> Rivermax - GPU direct, Communicating between nodel without passing buffer to CPU and system RAM</li> 
            </div>

            <div class="6u 12u$(medium)">            
                <span class="image fit"><img src="{% link assets/images/overall.png %}" alt="" /></span>
            </div>

        </div>
        
    </div>
</section>
  

  
