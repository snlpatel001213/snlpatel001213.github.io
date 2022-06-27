---
title: Patents, Certificate & Research Papers
layout: landing
description: 'Publications is a core part of what I do in day-to-day work.'
image: assets/images/pic07.jpg
nav-menu: true
---

<!-- Main -->

<!-- One -->
<div class="box" >
	<h2>Patents</h2>	
</div>
<section id="one">
	<div class="inner">
		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" >
						<a href="https://patents.google.com/patent/US20200311345A1/en?oq=20200311345" target="blank_">
							<img src="{% link assets/images/US20200311345A1.png %}" alt="" style=""/>
						</a>
					</span>
			</div>
			<div class="8u 8u$(small)">
				<b>System and Method for Language-independent Contextual Embedding</b><br>
				<b><a href="https://patents.google.com/patent/US20200311345A1/en?oq=20200311345">US US20200311345 · Issued Nov 9, 2021</a></b>
				<p>This patent covers the following functionality:
					<li> Character-based language independent embeddings generation in an unsupervised manner</li>
						<li> Multilingual alignment through custom loss function</li>
							<li> Using Skip connection for easing gradient propagation</li>
					<li> Snapshot ensemble technique to retrieve multiple models in a single run.</li>
				   Technology/logic used:PyTorch, Snapshot ensemble, Skip connection, Random multimodel CNN-LSTM ensemble
				</p>
			</div>            
		</div>

		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" >
						<a href="https://patents.google.com/patent/US20200311345A1/en?oq=20200311345" target="blank_">
							<img src="{% link assets/images/US20210034621A1.png %}" alt="" style=""/>
						</a>
					</span>
			</div>
			<div class="8u 8u$(small)">
				<b>System and method for creating database query from user search query</b><br>
				<b><a href="https://patents.google.com/patent/US20200311345A1/en?oq=20200311345">US US20210034621A1 · Filed Jul 30, 2019</a></b>
				<p>This is in natural language query (NLQ) domain. It basically comprise of way to build a query from natural language. 
				It classifies the question in to wheather it is single word answer or elaborated one. Then it generate search query which can be used to query database. 
				The patent also describes interplay of various modules such as named entinty resolution, question answering system nd abstractive summarization.
				</p>
			</div>            
		</div>

		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" >
						<a href="https://patents.google.com/patent/US20200104359A1/" target="blank_">
							<img src="{% link assets/images/US20200104359A1.png %}" alt="" style=""/>
						</a>
					</span>
			</div>
			<div class="8u 8u$(small)">
				<b>System and method for comparing plurality of documents</b><br>
				<b><a href="https://patents.google.com/patent/US20200104359A1/">US US20200104359A1 · Issued Dec 28, 2021</a></b>
				<p>This patent covers the following functionalities : 
					<li>Comparing documents semantically </li>
					<li>Comparison tools that show addition modification and deletions.</li>
					<li>Using Skip connection for easing gradient propagation</li>
					<li>Works on sentence and paragraph level</li>
					Technology used: Custom Sentence Vectorizer (Modified InferSent), Siamese Network
				</p>
			</div>            
		</div>

		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" >
						<a href="https://patents.google.com/patent/US20200104359A1/" target="blank_">
							<img src="{% link assets/images/US20200105379A1.png %}" alt="" style="height:20%"/>
						</a>
					</span>
			</div>
			<div class="8u 8u$(small)">
				<b>System and Method of Documenting Clinical Trials</b><br>
				<b><a href="https://patents.google.com/patent/US20200105379A1">US US20200105379A1 · Issued May 3, 2022</a></b>
				<p>This patent covers the following functionality: 
					<li>Differentiating primary and secondary Publicationsrelated to Clinical Trial </li>
					<li>Associating Clinical Trials to the Journal Publications based on the content of Clinical Trial and Publication</li>
					Technology used: Siamese Networks for with Convolution and Recurrent components, Attention based text classifiers. 
				</p>
			</div>            
		</div>
	</div>
</section>

<div class="box" >
		<h2>Book</h2>	
</div>
<section id="one">
		<div class="inner">
			<div class="row">
				<div class="4u 4u$(medium)" >
						<span class="image fit" ><a href="https://www.amazon.com/dp/9389898110/" target="blank_"><img src="{% link assets/images/book.png %}" alt="" /></a></span>
				</div>
				<div class="8u 8u$(small)">
					<b>Getting started with Deep Learning for Natural Language Processing</b>
					<p>This book covers wide areas, including the fundamentals of Machine Learning, Understanding and optimizing Hyperparameters, Convolution Neural Networks (CNN), and Recurrent Neural Networks (RNN). This book not only covers the classical concept of text processing but also shares the recent advancements. This book will empower users in designing networks with the least computational and time complexity. This book not only covers basics of Natural Language Processing but also helps in deciphering the logic behind advanced concepts/architecture such as Batch Normalization, Position Embedding, DenseNet, Attention Mechanism, Highway Networks, Transformer models and Siamese Networks. This book also covers recent advancements such as ELMo-BiLM, SkipThought, and Bert. This book also covers practical implementation with step by step explanation of deep learning techniques in Topic Modelling, Text Generation, Named Entity Recognition, Text Summarization, and Language Translation. In addition to this, very advanced and open to research topics such as Generative Adversarial Network and Speech Processing are also covered.</p>
				</div>            
			</div>
		</div>
</section>

<div class="box" >
	<h2>Research Papers</h2>	
</div>
<section id="one">
	<div class="inner">
		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" ><a href="https://ieeexplore.ieee.org/document/9434169" target="blank_"><img src="{% link assets/images/pub/1.png %}" alt="" /></a></span>
			</div>
			<div class="8u 8u$(small)">
				<b>Sample Specific Generalized Cross Entropy for Robust Histology Image Classification</b><br>
				<p>2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI) · Apr 15, 2021</p>
				<p>The accuracy of deep learning classifiers trained using the cross entropy loss function suffers even when a fraction of training labels are wrong or input images are uninformative. 
					We take advantage of the bootstrapping properties in deep learning models in order to design loss functions that are aware of the difficulty of classifying individual samples, due to either the label noise or the lack of a strong visual signal. We carry out extensive experiments to validate our approach by comparing against the models trained using other loss functions. 
				</p>
			</div>            
		</div>
	</div>
</section>
<section id="one">
	<div class="inner">
		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" ><a href="https://ieeexplore.ieee.org/document/9434121" target="blank_"><img src="{% link assets/images/pub/2.png %}" alt="" /></a></span>
			</div>
			<div class="8u 8u$(small)">
				<b>Fast, Self Supervised, Fully Convolutional Color Normalization Of H&E Stained Images</b><br>
				<b>2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI) · Apr 15, 2021</b>
				<p>Performance of deep learning algorithms decreases drastically if the data distributions of the training and testing sets are different. Due to variations in staining protocols, reagent brands, and habits of technicians, color variation in digital histopathology images is quite common. We propose a color normalization technique, which is fast during its self-supervised training as well as inference. Our method is based on a lightweight fully-convolutional neural network and can be easily attached to a deep learning-based pipeline as a pre-processing block.</p>
			</div>            
		</div>
	</div>
</section>
<section id="one">
	<div class="inner">
		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" ><a href="https://deepai.org/publication/a-study-of-traits-that-affect-learnability-in-gans" target="blank_"><img src="{% link assets/images/pub/3.png %}" alt="" /></a></span>
			</div>
			<div class="8u 8u$(small)">
				<b>A study of traits that affect learnability in GANs</b><br>
				<b>2021 2nd International Conference on Computer Vision, Communications and Multimedia (CVCM 2021) · Nov 27, 2020</b>
				<p>Generative Adversarial Networks GANs are algorithmic architectures that use two neural networks, pitting one against the opposite so as to come up with new, synthetic instances of data that can pass for real data. Training a GAN is a challenging problem which requires us to apply advanced techniques like hyperparameter tuning, architecture engineering etc. Many different losses, regularization and normalization schemes, network architectures have been proposed to solve this challenging problem for different types of datasets. It becomes necessary to understand the experimental observations and deduce a simple theory for it. In this paper, we perform empirical experiments using parameterized synthetic datasets to probe what traits affect learnability.</p>
			</div>            
		</div>
	</div>
</section>
<section id="one">
	<div class="inner">
		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" ><a href="" target="blank_"><img src="{% link assets/images/pub/4.png %}" alt="" /></a></span>
			</div>
			<div class="8u 8u$(small)">
				<b>DeepInteract: Deep Neural Network Based Protein-Protein Interaction Prediction Tool</b><br>
				<b>Current Bioinformatics · Jan 1, 2017</b>
				<p>Using Deep neural network to predict interaction between proteins. State of the art work accurately predicts interactions with 95+% accuracy in 5 organiams</p>
				<p>Live server : <a href="https://bioserver.iiita.ac.in/deepinteract/" target="blank_">DeepInteract</a></p>
			</div>            
		</div>
	</div>
</section>
<section id="one">
	<div class="inner">
		<div class="row">
			<div class="4u 4u$(medium)" >
					<span class="image fit" ><a href="https://link.springer.com/article/10.1007/s13721-016-0129-2" target="blank_"><img src="{% link assets/images/pub/5.png %}" alt="" /></a></span>
			</div>
			<div class="8u 8u$(small)">
				<b>DeepLNC, a long non-coding RNA prediction tool using deep neural network</b> <br>
				<b>Springer - Network Modeling Analysis in Health Informatics and Bioinformatics volume 5, Article number: 21 (2016)</b>
				<p>The significant role of long non-coding RNAs (lncRNAs) in various cellular functions, such as gene imprinting, immune response, embryonic pluripotency, tumorogenesis, and genetic regulations, has been widely studied and reported in recent years. In this paper we use deep neural network to predict if the given sequence is long non coding RNA</p>
				<p>Live server : <a href="https://bioserver.iiita.ac.in/deeplnc/index.php" target="blank_">DeepInteract</a></p>

			</div>            
		</div>
	</div>
</section>
  
