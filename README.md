# X-Twitter-Sentiment-Analysis
#**X-Twitter Sentiment Analysis Using Machine Learning, Deep Learning, and Transformers**


##**Overview**


Twitter, one of the world's most dynamic social media platforms, enables users to freely express opinions on a variety of topics. While this freedom empowers independent voices, it also introduces the risk of misinformation and emotional bias. To better understand public sentiment, especially in sensitive areas, sentiment analysis of tweets becomes crucial.

This project focuses on building models that classify tweets into three categories — Positive, Negative, or Neutral — using a range of methods:

Machine Learning: Support Vector Machines (SVM), Naive Bayes

Deep Learning: Long Short-Term Memory networks (LSTM)

Transformer Architectures: BERT and RoBERTa

##**Objective**


Our primary goal is to develop models that can predict the sentiment of tweets with high accuracy.
This technique can be applied in real-world scenarios such as:

Gauging public reception of new products.

Analyzing political opinion trends.

Monitoring brand reputation.

##**Datasets**



Tweets mentioning India's Prime Minister.

Dataset Size: 162,981 tweets

Twitter US Airline Sentiment Dataset

Passenger tweets about airlines from February 2015.

Dataset Size: 14,000 tweets


##**Approach**


We experimented across three major branches:

###**1. Machine Learning**

Support Vector Machine (SVM):
SVM finds the optimal hyperplane to separate classes in a high-dimensional space. It’s widely recognized for its effectiveness in classification problems.

Naive Bayes Classifier:
A simple yet powerful probabilistic classifier based on Bayes' theorem. It assumes independence among predictors, making it computationally efficient.

###**2. Deep Learning**

Long Short-Term Memory Networks (LSTM):
LSTM, a specialized type of RNN, is designed to capture long-range dependencies in sequential data, avoiding issues like vanishing gradients. Perfect for time-series and text data like tweets.

###**3. Transformer-Based Models**

BERT (Bidirectional Encoder Representations from Transformers):
BERT introduces deep bidirectional understanding of language context. It was pretrained on massive corpora like the Toronto Book Corpus and Wikipedia, and can be fine-tuned for a variety of tasks including sentiment classification.

RoBERTa (Robustly Optimized BERT Approach):
An enhanced version of BERT with improved training strategies such as dynamic masking and larger mini-batches. It refines performance across multiple NLP benchmarks.

##**Results**



| Model                  | Accuracy (%) | Dataset Size |
|-------------------------|--------------|--------------|
| SVM                     | 85.47        | 162,981      |
| Naive Bayes             | 74.25        | 162,981      |
| LSTM (20 epochs)        | 83.71        | 14,000       |
| BERT (3 epochs)         | 97.61        | 162,981      |
| RoBERTa (3 epochs)      | 94.19        | 162,981      |


##**Conclusion**


BERT demonstrated outstanding performance and remains the most reliable model among all tested.

RoBERTa also performed well overall, but exhibited inconsistencies, particularly in classifying negative sentiments.

SVM, Naive Bayes, and LSTM provide good baseline results and are significantly faster to train compared to transformer models.

Each approach presents a tradeoff between training time, computational resources, and accuracy. In critical applications, transformer models are recommended due to their superior performance.

