# 100DaysOfML

Day 1: July 21, 2018.

Today's Progress: Introduction to ML/AI. 

Artifical Intelligence: The study of how to make computers perform functions which at present humans are good at. 
Examples: Voice and Speech Recognization, Face recognization and face identofication, Object detection, Intuition, Inferencing, Learning new skills, Decision making, Abstract thinking 

Two types of AI - Artificial Intelligence:
  Strong AI, Full AI or General Intelligence
  Narrow AI, Weak AI or Applied AI(AAI)

Some terms in AI: 
  Artificial Intelligence, Machine Learning, Data Mining, Big Data Analytics, Knowledge Discovery from Databases, and Data Science.
 
  ML is alogirthm part of AI, ML is set of algorithms that are implemented to achieve the intelligence. 
  Data Mining is collection of the steps, Collect the data, clean the data, format the data, analyize the data, apply algorithm on that data, test the algorithm, use the the data. 
  
Weak AI Examples: Amazon recommendtaions, Amazon's 'echo dot' and 'echo plus' voice recognition systems, telsa auto pilot, nissan pro assist auto, Pandora Internat Radio, Walmart Robots working store aisles - checking stocks
Strong AI: we don't have any Strong AI products yet. Broad AI is the ability to minic the cognitive functions of humans. 

Thougths: Its good to start reading about ML/AI


Day 2: July 22, 2018.

Today's Progress: Machine Learning Introduction
"A Computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with Experience E." - Tom Mitchell. 

A domain of ML is related to question of how to build computer programs that by themselves(automatically) improve or LEARN from experience. 

The languages are popularing for ML in these days are Lua, clozure, and Scala. 

Day 3: July 23, 2018.

Today's Progress: 

Machine Learning - Terminology:
Features - The number of unique or distinct propertie that are assoicated with each item of data is called as a feature.
Samples - One item in the data is called a sample, a datum.
Feature Vector - The is n-dimensional data that is indicative of the features that we have extracted and which belong to the dataset.
Feature Extraction - This is done when preparing the feature vector. It transforms data from dimensional space to a fewer dimensions.
Training/Evolution set - The set of data that is used during the training phase, this is the phase where our algorithm learns from the data. 

Types of Learning:
Supervised Learning, Unsupervised Learning, Semi-supervised Learning, Reinforcement Learning.

Supervise Learning: This technique used in Prediction. Examples are Classification and Regression.
Regression is predicting a continuous value.

Unsupervised Learning: This technique used in Analysis. The examples for Un-supervised learning are clustering, density estimation, dimensionality reduction, association analysis and hidden markov models. 

Semi-supervised Learning: 
Why do we need semi-supervised learning:
  1. Label data is very hard to obtain and in most cases its costly. We need experts to label the data.
      Speech Processing - For example if we want to get the switch board data set we need nearly 450 hours of annotation time for 1 hour of sppech.
  2. Unlabled data is cheap. 
  
Reinforcement Learning: In AI, Reinforcement learning can be considered as a kind of dynamic programming. The training happens here is in the form of reward and punishment. 

Examples of Reinforcement Learning: Learing to fly a helicopter, learning to play chess, learning to backgammon, learning to play alpha go, Robot floor cleaner etc.

Artificial Neural Networks and deep learning:
A computational model that is based on functions and structure of neurons in our brain. We provide information through the input nodes of network which goes through network. This affects the structure of the ANN, in the sense neural network changes - or learns based on the input we give and outputs its proceduces.
Using Artificial Neural networks, we can find complex non-lineral relationships between the input and output. We could create a model and patterns can be found in the data. So ANNs are in a way non-lineral statistical modeling paradigm.
In this context we can see deep learning as a technique where we have two or more than layers in the neural network. 

Thougths: Its good to know about general terms and definitions in ML/AI.


Day 4: July 25,2018.

Todays Progress: Read about Spark streaming module

Hadoop - A distributed computing framework to process millions of records. Hadoop consists two parts, HDFS and MapReduce.
HDFS - A file system to manage the storage data. 
MapReduce - A framework to process data across multiple servers. MapReduce is fundamentally a batch processing operation. The MapReduce framework does not allow real-time processing of data. 

Spark for real-time processing: A Banking Web site may need real time monitoring to get fraudulent transactions, daily reports, failures, timeout, errors. 

In case of above scenario, MapReduce does not work we need Spark. 

Apache Spark is general purpsoe engine to solve a variety of data processing problems. 
Spark has specific modules for specific cases - Spark SQL, MLLib, GraphX, Spark Streaming. 

Spark offers the Spark Streaming module for Real-time processing. This makes better alternative to Hadoop when manuplating data streams.


Day 5: July 26, 2018. 
Today's Progress: Applying Machine Learning Algorithms on DStreams

-Read about how the k-means clustering algorithm used to find patterns. 
-Read about applying k-means clustering to streaming data.
-Saw demo implement the algorithm in python on a real world twitter dataset to determine tweet location patterns using 1) spark steraming, 2) MLLib
-Read about Decay factor and halflife which let you tweak the forgetfulness of the algorithm
