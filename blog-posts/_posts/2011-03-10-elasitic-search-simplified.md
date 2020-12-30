---
layout: post
title: Working with Elasticsearch - Simplified
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2017-07-17
category: articles
tags: [Elastic Search]
img: elasticsearch.png
comments: true
share: true
---
Required Data File and Code script are present at my [GitHub](https://github.com/snlpatel001213/algorithmia/tree/master/elasticsearch) Repository.

​
Nowadays, search stack is the most required tool for any industry. With the explosion of data, it has become necessary for any industry to keep data in the searchable form. When it comes to search technology, two guys are well known in the market; 1) [Solr](http://lucene.apache.org/solr/) 2) [Elasticsearch](https://www.elastic.co/). Practically both performed similar work. I am comfortable with both of them but  Elasticsearch is more friendly and easy to learn.

Learning new things is a chaotic phase. In this phase, if a positive catalyst is provided then this process can be made easier and faster. In this tutorial, I will walk you through installation, query and getting result from Elasticsearch.

1. **Installation**

    I don't want to rewrite things which are available in its best form, please go through the relevant link and proceed with Elasticsearch installation.

    _Ubuntu_ :  [https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-elasticsearch-on-ubuntu-14-04](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-elasticsearch-on-ubuntu-14-04)

    _Windows_ : [https://www.elastic.co/guide/en/elasticsearch/reference/current/windows.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/windows.html)

    _Centos_ : [https://www.elastic.co/guide/en/elasticsearch/reference/current/rpm.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/rpm.html)

    _Mac_ : [https://chartio.com/resources/tutorials/how-to-install-elasticsearch-on-mac-os-x/](https://chartio.com/resources/tutorials/how-to-install-elasticsearch-on-mac-os-x/)

2. **Applicability**

    Elasticsearch can be used for any kind of data. You may use Elasticsearch search for to:
    1. Store patent documents where patent number serve as "id" and entire document serve as data.
    2. It can be used to store groceries item information for a retail store where a barcode number serves as "id" and another attribute of the product such as name, price and % discount constitute data portion.
    3. It can be used to store Pincode of any place with country, city, longitude, and latitudes as extra information.

3. **Data**

    We will go ahead with third one, pin code to city mapping case. I have downloaded some data about cities of USA.
    Data is having following attribute for each  city/area : Zipcode, ZipCodeType, City, State, LocationType, *Lat, Long, Location, Decommisioned, TaxReturnsFiled, EstimatedPopulation, TotalWages.* Out of these all attributes, we will consider only *Zipcode, city, state lat {latitude} and log {longitude}* for present use case.

4. **Indexing**

    Elasticsearch is a server, we need some client to inject data into this server. Practically you can use various clients in Java Python and many other languages. You may query Elasticsearch using curl also. I will be using Python again. There is a package in Python called “Elasticsearch” which can be used as a client.

    A. *Installing required packages*

    1. Elasticsearch

        `pip install elasticsearch`

    2. Requests

        `pip install requests`

    B. *Indexing using Python*

    A simple 10 line of code does the same for us

    ```python
    import requests
    import traceback
    from datetime import datetime
    from elasticsearch import Elasticsearch

    # by default we connect to localhost:9200
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    req = requests.get('http://localhost:9200')


    def IndexData(filename):
        """
        FileName where data exists and to be indexed to Elastic Search
        :param filename:
        :return: nothing; index data in Elastic search server
        """
        fileAsArray = open(filename, "r").read().splitlines()
        for eachlineNo in range(1, len(fileAsArray)):  # as first line is header, omitted
            try:
                print "Processing Line : ", eachlineNo
                Zipcode, ZipCodeType, City, State, LocationType, Lat, Long, Location, Decommisioned, TaxReturnsFiled, EstimatedPopulation, TotalWages = \
        fileAsArray[eachlineNo].split(",")
                es.index(index='usa-index', doc_type='usa', id=eachlineNo,
                        body={'Zipcode': Zipcode, 'City': City, 'State': State, 'Lat': Lat, 'Long': Long})
            except:
                print "Some Error Exist at Line : ", eachlineNo
    # to index data in Elastic Search
    IndexData("free-zipcode-database-Primary.csv")

    ```

    Here we have to these lines well
    ```python
    es.index(index='usa-index', doc_type='usa', id=eachlineNo, body={'Zipcode':Zipcode,'City': City, 'State': State, 'Lat': Lat, 'Long': Long})
    ```

    We have defined an index name "**usa-index**" and doc type "**usa**". Index can be compared with a name of a book, under which many chapters can be there, just like doc_type. Each such doc_type store specific data. If we have collected data about the entire world then the index can be written as "world" and individual doc_type for each "country" can be made. Such hierarchy speeds up data lookup.

    Here we have taken line number as id and rest of the data is given in form of JSON to index.
    This script will take around 5 minutes to index entire data into server, next come query.

5. **Query**

    Any client in irrespective any language can make query to server, we will consider few examples
    1. web browser

       The web browser can make a query by hitting proper URL to the server. Let's say we have Zipcode and we want to know more about it, we query in following way:
    [http://172.21.10.13:9200/usa-index/_search?q=Zipcode:611](http://172.21.10.13:9200/usa-index/_search?q=Zipcode:611)

        You will get following JSON as server response. JSON response clearly says the Zipcode is for city ANGELES of state PR.

        ```json
        {
            "took": 2,
            "timed_out": false,
            "_shards": {
                "total": 5,
                "successful": 5,
                "failed": 0
            },
            "hits": {
                "total": 1,
                "max_score": 7.1147695,
                "hits": [{
                    "_index": "usa-index",
                    "_type": "usa",
                    "_id": "3",
                    "_score": 7.1147695,
                    "_source": {
                        "Lat": "18.28",
                        "City": "ANGELES",
                        "State": "PR",
                        "Zipcode": "611",
                        "Long": "-66.79"
                    }
                }]
            }
        }
        ```
        simillarly any one can query with city and get all its zipcode all togather

        [http://172.21.10.13:9200/usa-index/_search?q=City:ANGELES](http://172.21.10.13:9200/usa-index/_search?q=City:ANGELES)

        ```json
        {"took":2,"timed_out":false,"_shards":{"total":5,"successful":5,"failed":0},"hits":{"total":99,"max_score":6.5361986,"hits":[{"_index":"usa-index","_type":"usa","_id":"3","_score":6.5361986,"_source":{"Lat": "18.28", "City": "ANGELES", "State": "PR", "Zipcode": "611", "Long": "-66.79"}},{"_index":"usa-index","_type":"usa","_id":"21485","_score":4.3909116,"_source":{"Lat": "48.02", "City": "PORT ANGELES", "State": "WA", "Zipcode": "98363", "Long": "-123.82"}},{"_index":"usa-index","_type":"usa","_id":"22098","_score":4.3909116,"_source":{"Lat": "33.97", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90001", "Long": "-118.24"}},{"_index":"usa-index","_type":"usa","_id":"22102","_score":4.3909116,"_source":{"Lat": "34.05", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90005", "Long": "-118.31"}},{"_index":"usa-index","_type":"usa","_id":"22111","_score":4.3909116,"_source":{"Lat": "34.04", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90014", "Long": "-118.25"}},{"_index":"usa-index","_type":"usa","_id":"22123","_score":4.3909116,"_source":{"Lat": "34.07", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90026", "Long": "-118.26"}},{"_index":"usa-index","_type":"usa","_id":"22169","_score":4.3909116,"_source":{"Lat": "34.09", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90072", "Long": "-118.3"}},{"_index":"usa-index","_type":"usa","_id":"22170","_score":4.3909116,"_source":{"Lat": "34.05", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90073", "Long": "-118.45"}},{"_index":"usa-index","_type":"usa","_id":"22174","_score":4.3909116,"_source":{"Lat": "34.1", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90077", "Long": "-118.45"}},{"_index":"usa-index","_type":"usa","_id":"22180","_score":4.3909116,"_source":{"Lat": "33.95", "City": "LOS ANGELES", "State": "CA", "Zipcode": "90083", "Long": "-118.39"}}]}}
        ```

    2. Python

        As elastic search package serve as a client, it can be used to query sever

        ```python
        def searchESforZipcode(zipcode):
             queryResult = es.search(index='usa-index', body={"query": {"match": {'Zipcode':zipcode }}})
            if queryResult['hits']['total'] >= 1:
                 return queryResult['hits']['hits'][0]['_source']
            else:
                 return False

            print (searchESforZipcode(611))
        ```

        *Response*:
        ```json
        {u'Lat': u'18.28', u'City': u'ANGELES', u'State': u'PR', u'Zipcode': u'611', u'Long': u'-66.79'}
        ````
        In above example, I have used the "match" as query type but you can use another query type also including term query, terms query, range query, regexp query etc..You may find more about query type from here

        By default elastic search return 10 search as result but you may modify that too by using extra parameter in query as size
        Example: [http://172.21.10.13:9200/usa-index/_search?q=Zipcode:611&size=100](http://172.21.10.13:9200/usa-index/_search?q=Zipcode:611&size=100)
        you may use another search modifier such as size, from, lenient, sort etc..You may find more about search modifier here.

        With this I am finishing this blog-post, I hope this will provide and clear and crisp guide about working elastic search. Experiment, explore and exploit with data to become the master
