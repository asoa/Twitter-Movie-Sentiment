#!/usr/bin/env python

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
from analysis.sentiment_analysis import SA
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import os
import random
import re
import nltk
from nltk.corpus import stopwords


class ModelComparison:
    def __init__(self, **kwargs):
        self.kwargs = {k:v for k,v in kwargs.items()}
        self.docs = []
        self.stopwords = stopwords.words('english')

        self.labels = np.zeros(4000)
        self.labels[2000:] = 1
        pos_files = random.sample(os.listdir('benchmark/pos'), 2000)
        neg_files = random.sample(os.listdir('benchmark/neg'), 2000)

        for fname in pos_files:
            with open('benchmark/pos/{}'.format(fname)) as f:
                no_stop = ' '.join([word for word in nltk.word_tokenize(f.read()) if word not in self.stopwords])
                self.docs.append(no_stop)

        for fname in neg_files:
            with open('benchmark/neg/{}'.format(fname)) as f:
                no_stop = ' '.join([word for word in nltk.word_tokenize(f.read()) if word not in self.stopwords])
                self.docs.append(no_stop)

    def vader_sa(self):
        sa = SentimentIntensityAnalyzer()
        vader_pred = []
        x_train, x_test, y_train, y_test = train_test_split(self.docs, self.labels, train_size=0.8, test_size=0.2, random_state=155)

        for review in x_test:
            sentiment = None
            score = sa.polarity_scores(review)
            pos = score['pos']
            neg = score['neg']
            neu = score['neu']
            cmp = score['compound']
            if cmp >= 0.05:  # positive
                sentiment = 0
                vader_pred.append(sentiment)
                if self.kwargs.get('debug'):
                    print(f"{review}: {sentiment}")
            else:  # negative
                sentiment = 1
                vader_pred.append(sentiment)
                if self.kwargs.get('debug'):
                    print(f"{review}: {sentiment}")
                # return 1
        print("***** Vader Classification *****\n")
        print(classification_report(y_test, vader_pred, target_names=['pos','neg']))
        print(confusion_matrix(y_test, vader_pred))

    def clean_movie_reviews(self, doc):
        # return [re.sub(r'\d+','', word) for word in doc if word not in self.stopwords and word.isalpha()]
        review = ' '.join([re.sub(r'\d+', '', word) for word in doc.split(' ') if word.isalpha()])
        if self.kwargs.get('debug') and self.kwargs.get('train'):
            print(review)
        return review

    def naive_bayes(self):
        cleaned_reviews = [self.clean_movie_reviews(doc) for doc in self.docs]

        print("***** Naive Bayes Training *****")
        tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', smooth_idf=True)
        tfidf = tfidf_vect.fit_transform([doc for doc in cleaned_reviews])
        tfidf = tfidf.toarray()
        tfidf = np.hstack((tfidf, np.reshape(self.labels, (tfidf.shape[0],1))))

        x_train, x_test, y_train, y_test = train_test_split(tfidf, tfidf[:,-1], train_size=0.8, test_size=0.2, random_state=155)
        print("***** Navie Bayes Classification *****\n")
        clf = MultinomialNB(alpha=1)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        # self.pred = clf.predict_proba(x_test)
        print(classification_report(y_test, pred, target_names=['pos','neg']))
        print(confusion_matrix(y_test, pred))


def main():
    mc = ModelComparison(debug=False)
    mc.vader_sa()
    mc.naive_bayes()


if __name__ == "__main__":
    main()
