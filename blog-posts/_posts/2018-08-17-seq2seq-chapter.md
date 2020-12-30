---
layout: post
title: Introduction to Sequence to Sequence learning
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2018-08-17
category: articles
tags: [sequence to sequence]
img: sequence_network_appliocation.png
comments: true
share: true
---

# How this blog series is different?
This is not just an ordinary blog a where I will show you some equation and sign off. This is an end to end blog series about Sequence to Sequence, We will start off with general Sequence to Sequence architecture, then one by one we will implement [Vanilla Encoder Decoder](https://arxiv.org/abs/1409.3215), [Attention](https://arxiv.org/abs/1706.03762), [Beam Search](https://guillaumegenthial.github.io/sequence-to-sequence.html), batch processing with attention, [Pointer Networks](https://arxiv.org/abs/1506.03134). A series of the blog where algorithm, equations, and Implementation will be parallelly covered. 

**Sequence to Sequence is one size fit all kind of algorithm which is used in all below given tasks:**

- Machine Translation
- Summarization
- Question Answering
- chit chatbot
- Text Simplification
- image to code
- Speech to text
- text to speech

<p align="center"><img class="img-responsive" src="../assets/img/sequence_network_appliocation.png"></p>
<p align="center">Figure 1. Symbolic representation showing different application of sequence to sequence network.</p>


I am very sure you will find hundreads of blogs with thsese topics, but there are very few which are actually helps in learning, many of them are just showing equations as copied from original papers. Many will just throw code and wont help in understanding given step was implemented, what equations are governing the flow. Many will show you implementation but you wont be able to reproduce or reuse that implementation in your pipeline. There can be bugs :bug:, errors or  questionable extensibility.
This series I will put my best so that you can actually correlate actual research with implementation and can further develope upon it. In present series we will be using PyTorch as is having intuitive model building approaches. It is easy to understand and debug code with PyTorch.
