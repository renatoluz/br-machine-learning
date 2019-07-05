# Nanodegree Engenheiro de Machine Learning
## Proposta de Projeto Final - Sentiment Analysis Applied to Stock Prediction
Renato Leal de Moura Luz  
05 de julho de 2019



### 1. Domain Background
Sentiment Analysis refers to the use of machine learning to identify the emotional reaction to an event, document or topic [1]. One of the possible applications of sentiment analysis is for predicting stock market movements. The internet is full of sources that represent the public opinion and sentiment about current events. Studies in [2] shows that the aggreagate public mood can be correlated with Dow Jones Industrial Average Index (DJIA).


### 2. Problem Statement

This capstone seeks a model which uses the top daily news headlines from Reddit ( /r/worldnews ) to predict stock market movement. A dataset with 8 years of daily news headlines and their respective DJIAs is available in Kaggle [3]. The stock market movement will be
modeled into a binary classification problem, where:

● 1 when DJIA Adj Close value rose or stayed as the same

● 0 when DJIA Adj Close value decreased .


### 3. Datasets and Inputs

Two channels of data are provided for this dataset:

1. Features: Historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date.

2. Target Variable: Stock data from Dow Jones Industrial Average (DJIA). The index is converted binary values where:

      ○ 1 when DJIA Adj Close value rose or stayed as the same

      ○ 0 when DJIA Adj Close value decreased.
  

Three data files are provided on Kaggle in .csv format:

1. RedditNews.csv: two columns The first column is the "date", and second column is the "news headlines". All news are ranked from top to bottom based on how hot they are. Hence, there are 25 lines for each date.

2. DJIA_table.csv: Downloaded directly from Yahoo Finance : check out the web page for more info.

3. Combined_News_DJIA.csv: This is a combined dataset with 27 columns. The first column is "Date", the second is the target variable (DJIA), and the following ones are news headlines ranging from "Top1" to "Top25".


The model will be implemented using only the file Combined_News_DJIA.csv . The range of the data is from 2008-06-08 to 2016-07-01 with a total of 3973 rows . The most recent two years of the dataset (about 20%), from 2015-01-02 to 2016-07-01, is going to be reserved for testing.


### 4. Solution Statement

First, the text data from the 25 features is going to be cleaned (some HTML tags are still present in the original data). Next, the text is going to be grouped and processed into feature vectors. The method bag of words [5] is going to be used to represent the text as numerical feature vectors. The bag of words model is going create a vocabulary of tokens from the headlines data and then counted. Also, the relevancy of words is going to be accessed using the method term frequency-inverse document frequency (tf-idf) [6]. Machine Learning algorithms from sklearn are going to be evaluated. The specific models are still going to be defined, but probably the first approach is going to be Logistic Regression and SVM (Stochastic Gradient Descent if it is too slow).


### 5. Benchmark Model

Since this dataset is from a Kaggle kernel, there is no ‘official’ benchmark available. Below are the scores from some Kaggle users who used the very same metric and test set that is going to be implemented in this project (AUC metric and 2 last years as test set):

User: Aaron7sun*
Method: CNN and LSTM
AUC score: 62-63%
Link: https://www.kaggle.com/aaron7sun/stocknews/discussion/23254

User: Kate
Method: Bernoulii Naive Bayes
AUC score: 59%
Link: https://www.kaggle.com/katerynad/bernoulli-naive-bayes-auc-59

User: Dan Offer
Method: Unknown
AUC score: 56%
Link: https://www.kaggle.com/aaron7sun/stocknews/discussion/23254

User: Kate
Method: Logistic Regression
AUC score: 49%
Link: https://www.kaggle.com/katerynad/logistic-regression

*This was the user who provided the original database

However, this benchmark will be used as a secondary benchmark. I have the following list to be considered as primary benchmark:

1. A ‘dummy’ classifier with random output

2. A dummy classifier with all 1 as output

### 6. Evaluation Metrics

The evaluation metric to be used is Area Under the Curve (AUC) which is a metric derived from receiver operating characteristic (ROC) curve [4]. The most recent two years of the dataset (about 20%), from 2015-01-02 to 2016-07-01, is going to be reserved for testing.

### 7. Project Design

Data -> Train/Test -> Data Preprocessing -> ML Algorithm -> Hyperparameter Optimization -> Report Test Score -> Repeat from ML Algorithm
