---
layout: post
title: Understanding Batching with Sequence To Sequence Architecture
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2018-02-10
category: articles
tags: [Machine Learning Advance, Python, LSTM]
img: understanding_seq2seq_batch_encoder.jpg    
comments: true
share: true
---


Lets say we want to build  a **"Word Doubler"**.  When given a sentence with 26 words it will convert it to 52 word sentence. In [Earlier](https://learningdeep.xyz/seq2seq/) Examples we have used batch size = 1 and hance only one example is processed at a time.  Earlier implementation was easy to implement but not efficient. To make it efficient we need to convert it to process $n$ example togather. This way we will be using rougly $n$ processor togather and will be processing $n$ time faster as well.

# Environment Setup

## Torch Installation

```bash
!pip install torch==0.4.1 -q
```

```python
from io import open
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F
```

For Demonstration we are taking batch size = 64. To efficiently use GPU each time we will be taking 64 sentences having length 26 and will convert all of them to 64 sentence having length 52 ($26*2$).


```python
hidden_size = 128
batch_size = 64
input_size = 25
output_size = 25
embed_size = 28
```

# Encoder

<p align="center"><img class="img-responsive" src="https://learningdeep.xyz/assets/img/understanding_seq2seq_batch_encoder.jpg"></p>


<p align="center">Figure 1. Illustration of how baching works with sequence to sequence. Present image shows an example where **(1)** Shows 64 sentence having fixed length of 26 words are processed in single batch. **(2)** Five example sentences are shown such as **"My name is Sunil"**, **"I'm Deep Learning Geek"**, **"I love CUDA"**, **"I love Pytorch"**,** "TF is Aww"**. To process them in batch each sentence is transposed. Each sentence is padded with "PAD" token to make legnth equal to **26** as shown in **(2)**. At each iteration at time *t*, row-wise 64 element are taken and 128 dimensional embedding is calculated for each word as shown in **(3)**. such 64 element are processed with LSTM/GRU as shown in **(4)**. LSTM/GRU results in encoder output and encoder hidden state , encoder hidden state will be used in *t + 1* **(5)**. Next time at *t + 1* iteration 64 elements are taken and processed and simillar way. This iteration will be repeated for 26 times and at last encoder hidden of *t = 26* is passed to Decoder to use it as Decoder hidden.</p>

Lets start with encoder, our encoder will use GRU (gated recurrent units).

Encoder unit takes two inputs, 1) input, and 2) Hiddden. Input Shape will be a tensor of size **[Batch_size, input_size]** -->** [64, 26]. For each batch row-wise 64 element are taken and 28 dimensional embedding is calculated.
** Hidden state shape will of size** [unidirectional, Batch_size , Hidden_size]** -->** [1, 64, 28]**

Here we are using trainable pytorch embeddings. Embed layer will convert each word in to fixed size vector, so for each batch embed layer will produce  **[Batch_size, input_size]** --> **[Batch_size, input_size, Embed_size]** . hence the ** [64, 1].** will be converted to ** [64, 1, 28].** (Each batch having 64 words, each word is represented by 28 dimentions)


```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).unsqueeze(1) #Input =  64 --->  #Output  [64,1 ]
        embedded = embedded.view(1, batch_size, embed_size) #Input = [64, 1]  --- > #Output =  [1, 64, 28]
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden #Output 1, 64, 128  #encoder Hidden = 1, 64, 128

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result #Output 1, 64, 128
```


```python
ENCODER = EncoderRNN(input_size,hidden_size)
```

I am now generating a fake dataset. of size [Batch_size, Input_size] --> [64, 26]. 

```python
input = []
for j in range(0,batch_size):
    temp = []
    for i in range(0, input_size):
        temp.append(random.randint(0, 5))
    input.append(temp)
```


```python
test_encoder_input = torch.tensor(input)
encoder_input_transposed = test_encoder_input.t()
```


```python
encoder_input_transposed.shape
```
>torch.Size([25, 64])

```python
encoder_hidden = ENCODER.initHidden(batch_size)
```


```python
for i in range (0,encoder_input_transposed.shape[0]):
    encoder_output, encoder_hidden = ENCODER(encoder_input_transposed[i],batch_size,encoder_hidden)
print("ENCODER OUTPUT SHAPE : ", encoder_output.shape, "ENCODER HIDDEN STATE SHAPE : ",encoder_hidden.shape)
```  
>ENCODER OUTPUT SHAPE :  torch.Size([1, 64, 128]) ENCODER HIDDEN STATE SHAPE :  torch.Size([1, 64, 128])


# Decoder

<p align="center"><img class="img-responsive" src="https://learningdeep.xyz/assets/img/understanding_seq2seq_batch_decoder.jpg"></p>


<p align="center">Figure 2. Where Decoding Phsase is shown.**(1)** Sentences having double lenght 52 (26*2) is shown with batch size 64. To make sentence equal padding (PAD token) is used as shown as **(2)**. Each time step slice of shape 64 is taken and 28 dimensional embedding is generated for eachword **(3)**. such slice havinf embedding having of shape [64, 1, 28] is processed with GRU/LSTM **(4)**. GRU/LSTM  gives two things as output, A. Decoder output and B. Decoder hidden **(5)**. Decoder outputs are stacked and will be final output after all time step (t = 52) are processed **(6)**. At each time step Decoder hidden od previous time step is used.</p>


Our Decoder will use GRU (gated recurrent units).

Decoder unit takes two inputs, 1) input, and 2) Hiddden. Input Shape will be a tensor of size **[Batch_size, input_size]** -->** [64, 52].** Hidden state shape will of size** [unidirectional, Batch_size , Hidden_size]** -->** [1, 64, 28]**

Last Hidden state of Encoder will be first hidden state of the Decoder. 

Here we are using trainable pytorch embeddings. Embed layer will convert each word in to fixed size vector, so for each batch embed layer will produce  **[Batch_size, input_size]** --> **[Batch_size, input_size, Embed_size]** . hence the ** [64, 1].** will be converted to ** [64, 1, 28].** (Each batch having 64 sentences, each word is represented by 28 dimentions)

You need to decode element by element for the mini-batches. The initial decoder state [batch_size, hidden_layer_dimension] is also fine. You just need to unsqueeze it at dimension 0, to make it [1, batch_size, hidden_layer_dimension].

Please note, you do not need to loop over each example in the batch, you can execute the whole batch at a time, but you need to loop over all batches 52 of the input of dim [64, 52].

```python
class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.2)

    def forward(self, input, batch_size, hidden,training=True):
        embedded = self.embedding(input)   #Input =  64 --->  #Output  [64,1]
        if training == True:
            embedded = self.drop(embedded)
        embedded = embedded.unsqueeze(1).view(-1, batch_size,embed_size)  # Input = 64, 1, 128  --- > #Output =  52, 64, 128
        output = embedded
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden #Output 26, 64, 128  #encoder Hidden = 1, 64, 128

    def initHidden(self, batch_size):
        result = Variable(torch.empty(1, batch_size, self.hidden_size, device=device))
        result = nn.init.xavier_normal_(result)
        return result
```


```python
DECODER = DecoderRNN(hidden_size,output_size)
```

I am now generating a fake dataset. of size [Batch_size, Input_size] --> [62,52]. 

```python
main = []
for j in range(0,batch_size):
    temp = []
    for i in range(0, output_size):
        temp.append(random.randint(0, 5))
    main.append(temp)
```


```python
test_decoder_input = torch.tensor(main)
decoder_input_transpose = test_decoder_input.t()
```

Each decoder output will be of [64, 25] for 52 batches. This represent distribution of probability for 25 words for batch size of 64. Argmax is applied to [64, 25] and it will select top 1 word with highest probability and the resultant output for each batch will be [64, 1].

Such [64,1] output for 52 baches will give output sentence of double size(52) [64,52].


```python
decoder_hidden = encoder_hidden
for i in range (0,decoder_input_transpose.shape[0]):
    decoder_output, decoder_hidden = DECODER(decoder_input_transpose[i],batch_size,decoder_hidden)   
```


```python
print("DECODER OUTPUT : ", decoder_output.shape,"  DECODER HIDDEN STATE :", decoder_hidden.shape)
```

>DECODER OUTPUT :  torch.Size([64, 25])   DECODER HIDDEN STATE : torch.Size([1, 64, 128])


# Loss function for Sequece to Sequence

In a batch all sequecne are not of same length. So we must not calculate loss for padding (OR PAD) tokens added to input batches. To avoid this, a masked loss is calculated. In addition to PAD sometime EOS (End of Sequence) is added. Normal NLLLoss is calculated and loss corresponding to PAD is made to zero by masking. Resultant loss will be equavalent to average of loss derived by deviding total loss by total non-PAD tokens.


```python
class customLoss(nn.Module):
    def __init__(self,tag_pad_token = 1):
        super(customLoss, self).__init__()
        self.tag_pad_token = tag_pad_token

    def forward(self,logits, target):  
        target_flat = target.view(-1)
        mask = target_flat > self.tag_pad_token
        loss = nn.NLLLoss(reduce=False)(logits,target)
        loss = loss*mask.float()
        result = loss.sum()/len(target)
        return result
```
