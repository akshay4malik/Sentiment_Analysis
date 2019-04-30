#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:54:16 2019

@author: akshay
"""

import pandas as pd
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")

def lemmatization(tweet_list):
   lemma_sents = []
   
   for sent in tweet_list:
      new_sent = ''
      nlp_sent = nlp(sent)
      for i in nlp_sent:
         a = i.lemma_
         new_sent = new_sent + ' ' + a
      lemma_sents.append(new_sent)
   
   return lemma_sents
         
   


def clean_tweets(tweet_list):
   clean_tweets = []
   for tweet in tweet_list:
      urls_oper = re.sub(r'http\S+', '', tweet)
      punct_oper = re.sub('[^A-Za-z0-9]+', ' ', urls_oper)
      clean_tweets.append(punct_oper)

   return clean_tweets   
   
def remove_stopwords(tweet_list):
   sents = []
   for twt in tweet_list:
      twt.lower()
      www = ""
      for i in twt.split():
         if i not in STOP_WORDS:
            www = www + ' ' +i
      sents.append(www)
   
   return sents
         
   
def main():
      
   file = pd.read_csv('train_2kmZucJ.csv')
   
   tweets = list(file['tweet'])
   
   cleaned_tweet = clean_tweets(tweets)
   cln_lst = remove_stopwords(cleaned_tweet)
   lemma_twts = lemmatization(cln_lst)
   Y = list(file['label'])
   
   
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(lemma_twts, Y, test_size = 0.2, random_state = 0)
   
   ## applying classification model
   
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
   
   dataset = pd.DataFrame({'text':cln_lst})
   ## using tf-idf vectorizer
   tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5500)
   tfidf_vect.fit(dataset['text'])
   xtrain_tfidf = tfidf_vect.transform(X_train)
   xvalid_tfidf = tfidf_vect.transform(X_test)
   from imblearn.over_sampling import SMOTE
   #from imblearn.under_sampling import RandomUnderSampler
   #rus = RandomUnderSampler(random_state=0)
   smt = SMOTE()
   #X_train_new, y_train_new = rus.fit_resample(X_train, y_train)
   X_train_new, y_train_new = smt.fit_sample(xtrain_tfidf, y_train)
   
   
   
   
     
#   classifier = RandomForestClassifier(class_weight={0:1,1:1}, criterion='entropy',
#               max_depth=5, max_features='sqrt',n_estimators=250, random_state=0).fit(X_train_new, y_train_new)
   '''
   classifier = RandomForestClassifier(criterion='entropy',
               max_depth=12, max_features='sqrt',n_estimators=3000).fit(X_train_new, y_train_new)
   predictions = classifier.predict(xvalid_tfidf)
   
   '''
   from sklearn.naive_bayes import MultinomialNB
   classifier = MultinomialNB(alpha=1.95).fit(X_train_new, y_train_new)
   predictions = classifier.predict(xvalid_tfidf)
   
   
   
   
   from sklearn.metrics import confusion_matrix
   from sklearn.metrics import f1_score
   from sklearn.metrics import accuracy_score
   from sklearn.metrics import roc_auc_score
   
   cm = confusion_matrix(y_test, predictions)
   f1_sc = f1_score(y_test, predictions)
   acc = accuracy_score(y_test, predictions)
   from sklearn.metrics import classification_report
   m =classification_report(y_test, predictions)
   print (m)                  
   print ('f1_score', f1_sc)
   print ('roc_score: ',roc_auc_score(y_test,predictions))
   print ("confusion matrix: ", cm)
   print ('accuracy :', acc)
   
   test_file = pd.read_csv('test_oJQbWVk.csv')
   test_twt = test_file['tweet']
   test_cleaned_tweet = clean_tweets(test_twt)
   test_cln_lst = remove_stopwords(test_cleaned_tweet)
   test_lemma_lst = lemmatization(test_cln_lst)
   test_x_tfidf = tfidf_vect.transform(test_lemma_lst)
   output = classifier.predict(test_x_tfidf)
   op_df = pd.DataFrame({'id':test_file['id'],'label':output})
   op_df.to_csv("result.csv")
   


   
if __name__ == '__main__':
   main()
   

