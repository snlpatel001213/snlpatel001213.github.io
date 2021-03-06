---
layout: post
title: "Understanding The Butterfly Effect – Market Intelligence"
img: market-intelligence.png # Add image post (optional)
date: 2017-07-12 12:55:00 +0300
description: # Add post description (optional)
tag: [Travel, Blogging, Mountains]
comments: true
share: true
---
“Money matters to all, the most”. I am working in Market intelligence for 1.5 years and today I am going to share some of the techniques for market research. Any market research task is a perfect example of big data processing. Hence market research can be characterized by five "V"s namely volume. veracity, velocity, variety, and Volatility.

## A. Volume ##

An indefinite amount of data is available for each segment of the market.
For example, if you look at this site of us government, you may find a dump of twitter data. Each days dump is around 45GB [huge huge huge amount of data]. All though share markets data is not available for public use but can be estimated of same volume each day.
To handle this amount of data from storing to pre-processing to analytics is a mammoth task. We will also see how one can design a robust system that can practically deal with such volume.

## B. Variety ##
Based on variety, data can be classified into three major type.

1. Structured
Structured data is one where data is generated by machines or from the source itself available in the proper format. Such data require a minimum amount of steps before preprocessing.
Example: Share market data. Clinical trial data[https://clinicaltrials.gov/]

2. Semi-Structured
Semi-structured data is having some part well structured and some part may be present in raw form
let's say one wants to keep watch on the activity and public profile of particular company or entity. Well, structured Public data about that companies growth can be obtained from Exchanges such as Nasdaq or BSE or NSE etc. Unstructured data for such entity/ company can be obtained from various sources such as news/ tweets. Unstructured data requires the great amount of preprocessing and to synchronize between an incoming stream of structured as well as unstructured data is the greater task to do. Usually, it happens that in a system structured data is presently ready to apply analytics form but without bringing unstructured data to the same stage, the decision cannot be made

3. Unstructured
Unstructured data usually included everything that is not structured, simple!!. Unstructured data comes many challenges and in this blog post, I will be practically explaining you, how to deal with such data.
We will see what are all steps and type of analytics can be applied to such data.

## C. Velocity ##
Perhaps this is just a repetition of what is said while discussing regarding the volume of the data. People are continuously generating data, be it with their mobile device or IOT device or chat services or video streaming or audio streaming or satellite data. For example, Twitter is dumping per day data of 45 GB. I have practically worked on a banking data stream of 20GB/day. According to an estimation, Every day 2.5 Quintillion bytes of data are created. It's tremendous!!. let's say out of this data only 1% is of our business then also is a huge amount of data to deal with.
<p align="center">
<img class="img-responsive" src="https://static.wixstatic.com/media/884a24_662a6388fadd46cabb383f9560c1e2c4~mv2.png/v1/fill/w_435,h_435,al_c,usm_0.66_1.00_0.01/884a24_662a6388fadd46cabb383f9560c1e2c4~mv2.png">
</p>
    _Figure 1. Showing overall data generation in one minute by at different portal_

## D. Veracity ##

Veracity actually refers to the part of the data which is actually of our interest, Person may have access to data from all public companies but on which company he is focusing is of more importance. To get data of our business from the 95% of irrelevant data is also a challenging task. One person may be interested in medical field related tweets but out of all dumps of twitter data how much is of use and how to extract this is much important.

## E. Volatility ##

When data is used for “Actionable intelligence”, it is required that as soon as such data is generated, should be consumed and intelligence should be derived. Example again is stock exchange data, it cost much to get live streaming of such data. BSE and NSE provide such live stream at a cost of INR 20,00,000 as per their official rates. As the huge amount of data keeps coming to our systems, it is important to note how much time such data should be discarded or after how much time such data is of no use. I gave you an example that live-feed of stock exchange data in India costs INR 20,00,000 and same data is provided by NSE and BSE after 15 minutes of delay cost almost negligible. It's always necessary to decide a limit when data is useful and can help in “Actionable intelligence” and then after it's of no use and should be discarded.

We are done with discussing nature of our big data in context with market intelligence. All attributes of data are of immense importance, ignoring one may cost you high.

In this blog post, our discussion will be mostly centered on how to deal with unstructured data. Unstructured data mostly comprises of text. Textual data is the way of communication for humans and hence mostly textual data is of use.

# AIM #

Our aim always remains always focused on an entity or group of entities which may help in bringing more profit or popularity in any sense. We may focus on the progress of some public companies for the stock benefit. One may focus on the internal development of a competitor. One country may keep watch over internal politics of others to gain monitory and we as influential benefits.

# Entity Recognition #

1. Overview

    As soon as we decide our subject of interest, we decide some entities or factor to keep watch over. Altering state of such entities/ factors may help us in providing “competitive intelligence”
An entity may be :

    1. A political party of the country or another country
    2. A company whose share value is of our interest
    3. A medicinal agent discovered for a particular disease
    4. The spread of disease and related medicinal sale
    5. Supply chain management
    6. Release of a new product in the market

    This is a much wider topic to deal with, almost all part of market research aims for actionable competitive intelligence.
Such entities of interest may of two type

    1. Definite: when we keep watch over definitely known entities. This makes our task simpler, few examples are :

        a) Name of all public companies registered in Nasdaq or NSE or BSE.
b) Narrowing our interest. Out of all politician in India, we keep watch over some of them. If we keep watch over all of them, we end up with an indefinite list and such list changes time by time.

    2. Indefinite (variable): keeping our scope limited (definite) may make our task easy but often do not provide information about factor may be passively influencing out task out of scope.
Few examples of indefinite scope are :

         a) Keeping watch over new medicine compound invented by some company for some disease. You may argue whats new in that because billions of compounds in the world exists and new is invented day by day. It's really hard to keep watch on. If we keep a list of such entities then one must iterate through all stored chemical in order to find out one present in news/tweet. Such loop iterate over billions of chemical name each time a tweet or newsfeed arrives in the system. One cannot keep watch over newly invented chemical compounds by keeping a stationery list.

        b) Name of a place where some incidence happened. The World is full of places and its never a good idea to keep a list of all places and apply string matching whenever a new news/tweet arrives in the system

2. Technology

    Now comes how to identify such entities to at least separate data of our use form the huge amount of streaming or stationary data.
Named entity recognition is a bigger task, and cannot be accomplished by string matching and regex because of following reasons:

    2.1. An entity may comprise of multiple words and to each such entity arrangement of words in data and in the list where it is present must be same. According to this approach, Barack Obama and Obama Barack will not match, think of such millions of names, then you will be able to understand the complexity we are talking about.

    2.2. Use of regex makes the overall system slower. it is practically impossible to design a regular expression which takes care of all wild cases.

    2.3. Although fuzzy searching with a threshold by applying distance measure algorithm is a little bit practical idea but do not provide required control.

    At the end we are left with one thing in hand, to design such a system which is having sufficient knowledge about domain, syntax, and grammar of the language to identify such entities without any predefined list. Such system should also be capable of predict new entities which were unknown at the time of training.
Named entity recognition(NER) is addressed by NLTK and SPACY like toolbox by providing NER for companies, locations, organizations, and products. In practice, this functionality never made for industrial grade NER and to achieve greater perfection one must train their own NER on custom data.
Named entity resolution mainly solved by the application of Conditional Random Field (CRF). A CRF is a discriminative, batch, tagging model, in the same general family as a Maximum Entropy Markov model.
What I have gained practically after using CRF for Named entity recognition is :

    1. CRF gives around 70% practical accuracy and its false positive rate is much higher.
    2. Such NER system if trained in ensemble with better deep learning techniques like LSTM and CNN it gives very good results

    Logically in an ensemble of CNN, LSTM and CRF each component is having its own purpose

    __1. CNN__
Convolution networks are made to identify visually similar features. Employing CNN will identifying the entity that may have resemblance to a previously exposed entity and hence explicit training with an entire list of entities is not required.
Example: A) Auckland of New Zealand and Oakland, California
B) Names of major antibiotics end with suffix “mycin” such chloromycin, streptomycin etc

    __2. LSTM__
LSTM is known for language processing, LSTM stands for Long Short-Term Memory. LSTM being a derived form of RNN, it takes care of inter-word dependencies.
For example: for to identify the name of companies, Microsoft corporation is given as one of the names while training, it will identify Deloitte Corporation also as company name as it would inherently make a rule that something before word corporation can be a name of the organization.
Bidirectional LSTM applies such logic in both the direction of the word; forward and backward direction so most robust results can be obtained.

    __3. CRF__ takes word level as well as an inter-word level feature in consideration.
If w1 w2 w3 are three consecutive words in a sentence then CRF takes following features to calculate the probability of w2 being an entity.
For w1 w2 and w3, it will generate the following feature:

    1. is capital – [Share name are written in the capital in general;l e.g. GOOG, MSFT]
    2. is small
    3. is title case [title case have the greater probability of being an entity
    4. is alphanumeric
    5. is number
    6. word length [bioplogical names are longer in general]
etc.. one may append such “n” number of relevant feature.

    An overall system for named entity recognition can be generalized as given in below flow chart.

    <p align="center">
    <img class="img-responsive" src="https://static.wixstatic.com/media/884a24_7efab8a3b2ab4a94badcd05f827e4830~mv2.jpg/v1/fill/w_719,h_445,al_c,q_80,usm_0.66_1.00_0.01/884a24_7efab8a3b2ab4a94badcd05f827e4830~mv2.webp">
    </p>
    _Figure 2. A named entity recognition system comprises of ensemble of 1) Convolution Network 2) LSTM and 3) CRF_

    To gain practical experience on how to design custom NER refer following links:

    1. https://nlp.stanford.edu/software/CRF-NER.shtml
    2. https://spacy.io/docs/usage/training-ner
    3. https://www.machinelearningpython.org/single-post/How-to-design-a-machine-learning-model

    System Architecture Note :

> Always design a loosely coupled system – never keep a fix list of entity/ entity recognizer in the main program, better prefer to have indexing service like ElasticSearch in docker like container. If you keep such static list of entities hardbound into main program file then to edit such list you need to stop the process. With Elastic search like server, you may edit your list and keeping the main program unaffected and will automatically start working on new entity list.

# Relationship Mining #

1. Overview

    Once we identify entities, second comes relations between two entities. I am going to quotes a well-known recent example here “trump2cash - A stock trading bot powered by Trump tweets”. The quoted article suggests an effect of trumps speech on the stock exchange with on such occasions actionable insight may benefit us. Here trump and most affected public share can be related and keep such historical incidence in mind profit can be gained.

2. Technology

    This task seems similar to the sentiment analysis. We can think of a single sentence where two entity are present and word connecting to them represent positive or negative effect.
Once we have our entity tagged per sentence we need to see what is the relation between them. We have to train a system which can identify a positive and negative relationship between two entities or event and entity. For this e can do two things

    1. Train a sentiment analyzer larger known pre-tagged sentences
    2. Generate our own training set: that sentence having positive words like (shoots high, Hits etc) between two entities mark them as 1 and those having negative words (bearish bullish, plummets etc) mark them as 0. train a system with such a training data set.
    3. Combine approach 1 and 2 for more robust results.
Instead of using n-gram or bag of word approach, try to go for more robust approach RNN and LSTM. This is about relationship mining when two things are present in one sentence, what is present in a different sentence? Let's see next

# Co-reference resolutions #

With more than 70% cases you will find two entity to be related is present in a different sentence with the same context. What to do then, here comes co-reference resolution. “Vishal Sikka put down his papers. His action made great fall in Infosys share price” here word “his” indirectly refers to “Vishal Sikka”. Although both things under consideration “Vishal Sikka” and “Infosys” are not in the same sentence but such sentence should also be taken care of. This is called as reference resolution. Co-reference resolution is itself a complex problem yet to be solved. I can think of only one possible approach, that is sentence parsing and resolution.
A freely available tool such as spaCy is able to parse grammar in the sentence with human-level accuracy. Once summer parsing is done, one can apply rules on such sentences.

# Clustering #
Clustering is essential for to understand “the butterfly effect”. According to this “when a butterfly moves its wings in some part of the world it can make a tornado in some other part of the world”. All entities are connected to all with larger or smaller probability. All those are well-connected falls in the same cluster and affect each other to a greater extent. “Narendra Modi and Indian GDP are closely associated entity”
The action of ones directly affects another. Such as:

1. The New York Stock Exchange opens for business at 9:30 a.m. EST each day. However, prior to the opening trade on the NYSE, equity markets in Asia and Europe have already (or almost) finished their trading day. The point is, if certain stocks and/or sectors have had a particularly good or bad day in those markets, the sentiment could have an impact on trading here in the U.S.
2. If there is talk that China may revalue its currency (the yuan), then it may cause shares of exporters to China to trade higher.
3. The Internet has transformed the way people invest, as well as the way the public at large obtains news. Therefore, if a Web writer or journalist disseminates a bullish or bearish article about a company throughout the trading day, this can have a huge impact on its stock.
Keeping watch over such complex and closely associated member in one cluster is essential. The bigger question is how to do such clustering? Yes, we will do it practically with news data.
I have used a simple but efficient technique to demonstrate clustering in entities. I have trained a word to vector model with news dumps (129 MB only) of Bloomberg and Reuters. The result I got, clearly indicates the presence of clusters. Results are as follows :

* I want to find out, what are other companies related to JP Morgan. It returned me following list of probabilities.

```python
[('jpmorgan,', 0.6153316497802734),
("jpmorgan's", 0.5839070081710815),
('chase', 0.5646278858184814),
('citigroup', 0.5529909729957581),
('citi', 0.5488744378089905),
('morgan', 0.5226957201957703),
('ubs', 0.5187379121780396),
('citigroup,', 0.5123351812362671),
('bear', 0.5042657852172852),
('wachovia', 0.5008693337440491)]
```

* I want to find out what are other companies related to Microsoft and Google. It returned me following list with probabilities.

```python
[('yahoo', 0.7744687795639038),
("google's", 0.7412586212158203),
('oracle', 0.7128864526748657),
('apple', 0.7068471908569336),
('facebook', 0.7065688371658325),
("microsoft's", 0.6984558701515198),
('hp', 0.6734104156494141),
('microsoft,', 0.6706593036651611),
('ebay', 0.6601837277412415),
('at&t', 0.6566265225410461)]
```
A nonspecific dataset of mere 129MB size can give this information, you may think what a specific dataset of size Gigabyte can give.

I am concluding this blog-post for now, I know it's not yet complete. I will be kept on updating it as an when I will finish with rest of the data.
