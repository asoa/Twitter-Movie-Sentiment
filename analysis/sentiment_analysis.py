#!/usr/bin/env python
import timeit

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streaming.stream import Stream
from util.helpers import repr_json
import json
from util.helpers import write_ndjson
import os
from nltk.corpus import movie_reviews
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import nltk
import string
import re
import numpy as np
import pickle
from collections import defaultdict
import traceback


class SA:
    def __init__(self, **kwargs):
        self.kwargs = {k:v for k,v in kwargs.items()}
        self.method = self.kwargs.get('method', 'nb')
        self.db_name = self.kwargs.get('db_name', 'tweets')
        self.coll_name = self.kwargs.get('coll_name')
        self.tweet_files_path = self.kwargs.get('tweet_files_path')
        self.root_path = self.kwargs.get('root_path')
        self.tweets = None
        self.s = Stream(db_name=self.db_name, coll_name=self.coll_name)  # use some of Stream methods
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.combined_array = None
        self.words = []

        if self.kwargs.get('train'):
            self.pprocess_induction()
            self.naive_bayes()

        if not self.kwargs.get('train') and not self.kwargs.get('vader', False):
            self.pprocess_induction()
            self.get_tweets()
            # array = t.pprocess_predict(string='cpt marvel was a really good movie')
            array = self.pprocess_predict()
            self.naive_bayes(array)
            self.write_json(f_name=os.path.join(self.kwargs.get('root_path'), 'output', 'nb_results.json'))

        if self.kwargs.get('vader'):
            self.get_tweets()
            # self.vader_sa()

    def get_tweets(self):
        # os.chdir(self.kwargs.get('tweet_files_path'))
        for file in os.listdir(self.tweet_files_path):
            f_name = os.path.basename(file)
            if file.endswith('.json'):
                with open(os.path.join(self.kwargs.get('tweet_files_path'),file), 'r') as f:
                    self.json_objects = (json.loads(tweet) for tweet in f.readlines())
                if self.kwargs.get('vader'):
                    self.tweets = [{'text': tweet['text'], 'place': tweet['place'], 'sentiment': self.vader_sa(tweet['text'])} for tweet in self.json_objects if tweet.get('place') is not None]
                    # self.tweets = [{'text': tweet['text'], 'sentiment': self.vader_sa(tweet['text'])} for tweet in self.json_objects]
                    self.write_json(f_name=os.path.join(self.kwargs.get('root_path'), 'output', 'vader_{}'.format(f_name)))
                else:
                    self.tweets = [tweet for tweet in self.json_objects]
        return self.tweets

    def clean_movie_reviews(self, doc):
        # return [re.sub(r'\d+','', word) for word in doc if word not in self.stopwords and word.isalpha()]
        review = ' '.join([re.sub(r'\d+','',word) for word in doc if word.isalpha()])
        if self.kwargs.get('debug') and self.kwargs.get('train'):
            print(review)
        return review

    def drop_junk_sentences(self, word):
        """
        Args:
            word: (str) word from sentences generator

        Returns: (str) word with punctuation/html/numbers removed

        """
        pattern = re.compile(r'''
                RT @\w+:\s+Create your Avengers Endgame.*
            |   Create your Avengers Endgame*
            |   Blah                                     
        ''', re.VERBOSE)

        clean_str = re.sub(pattern, '', word)
        if len(clean_str) >= 3:
            return True
        return False

    def clean_tweets(self, doc):
        """ remove punctuation and stopwords """
        tknzr = nltk.TweetTokenizer()
        if self.kwargs.get('vader'):
            words = ' '.join([word for word in tknzr.tokenize(doc['text']) if word.isalpha()])
            return words
        else:
            words = ' '.join([word for word in tknzr.tokenize(doc)
                              if word.isalpha() and word not in self.stopwords and len(word) > 2])
            self.words.append(words)
            if self.kwargs.get('debug'):
                print(words)
            return words

    def create_array(self, tfidf, label=None):
        """ return numpy ndarray with column label """
        array = tfidf.toarray()
        # print(array.shape)
        # print(pos_array[:, -1])
        array = np.hstack((array, np.zeros((array.shape[0], 1))))  # add target column to the end of array
        # print(array.shape)
        array[:, -1] = label
        # print(array[:, -1])
        return array

    def pprocess_induction(self):
        # get combined vocabulary for pos and neg class
        all_reviews = (self.clean_movie_reviews(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids())
        if self.kwargs.get('stopwords', False):
            print("***** Stopwords kept *****")
            c_vocab_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
        else:
            print("***** Stopwords removed *****")
            c_vocab_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english')
        c_tfidf = c_vocab_vect.fit_transform([doc for doc in all_reviews])
        self.c_vocab = c_vocab_vect.get_feature_names()

        if not self.kwargs['train']:  # only get the combined vocabulary, no need to do tfidf logic
            return
        else:
            print("***** Training *****")
            # segment the movie reviews -> pos, neg and call the clean function to remove
            pos_documents = (self.clean_movie_reviews(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('pos'))
            neg_documents = (self.clean_movie_reviews(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('neg'))

            # fit and transform the the documents to a tfidf matrix using the combined vocabulary
            if self.kwargs.get('stopwords', False):
                print("***** Stopwords kept *****")
                tfidf_pos_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), smooth_idf=True, vocabulary=self.c_vocab)
                tfidf_neg_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), smooth_idf=True, vocabulary=self.c_vocab)
            else:
                print("***** Stopwords removed *****")
                tfidf_pos_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english', smooth_idf=True, vocabulary=self.c_vocab)
                tfidf_neg_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english', smooth_idf=True, vocabulary=self.c_vocab)
            pos_tfidf = tfidf_pos_vect.fit_transform([doc for doc in pos_documents])
            neg_tfidf = tfidf_neg_vect.fit_transform([doc for doc in neg_documents])
            # with open('vocab', 'wb') as f:
            #     pickle.dump(c_vocab, f)

            # combine both pos neg arrays into combined sparse matrix
            pos_array = self.create_array(pos_tfidf, 0)
            neg_array = self.create_array(neg_tfidf, 1)
            self.combined_array = np.concatenate((pos_array, neg_array), axis=0)
            # print(self.combined_array.shape)
            # print(combined_array[:1000,-1])  # should print all 0, pos reviews
            # print(combined_array[1000:, -1])  # should print all 1, neg reviews

            # print(f"pp_induct: {len(self.c_vocab)}")
            return self.combined_array

    def pprocess_predict(self, input_file=None, string=None):
        """
        import nltk movie review, create tfidf matrix with stopwords removed
        movie review documents are already tokenized

        Args:
            input_file: (str) filename with tweets
        """
        print("***** Predicting *****")
        # print(f"pp_predict: {len(self.c_vocab)}")
        if self.kwargs.get('stopwords', False):
            tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), smooth_idf=True, vocabulary=self.c_vocab)
        else:
            tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english', smooth_idf=True, vocabulary=self.c_vocab)
        # with ExitStack() as stack:
        #     path = '/Users/asoa/PycharmProjects/688/final_project/movie_sentiment/tweet_dir'
        if string:
            tweets = [self.clean_tweets(string).lower()]
        else:
            tweets = [self.clean_tweets(tweet['text'].lower()) for tweet in self.tweets]
            # files = [stack.enter_context(open(os.path.join(path,fname))) for fname in os.listdir(path)]
            # tweet_tfidf = tfidf_vectorizer.fit_transform([file.read() for file in files])
        tweet_tfidf = tfidf_vectorizer.fit_transform([tweet for tweet in tweets])
        tweet_array = tweet_tfidf.toarray()
        # print(f"tweet shape: {tweet_array.shape}")

        return tweet_array
        # print(tfidf_vectorizer.get_feature_names()[:100])

    def vader_sa(self, text):
        sa = SentimentIntensityAnalyzer()
        # if string is None:
        #     # tweets = [self.clean_tweets(tweet) for tweet in self.tweets]
        #     pass
        # else:
        #     tweets = string
        # for tweet in tweets:
        sentiment = None
        score = sa.polarity_scores(text)
        pos = score['pos']
        neg = score['neg']
        neu = score['neu']
        cmp = score['compound']
        if cmp >= 0.05:  # positive
            sentiment = 0
            if self.kwargs.get('debug'):
                print(f"{text}: {sentiment}")
            return 0
        elif (cmp < 0.05) and (cmp > -0.05):  # neutral
            sentiment = 2
            if self.kwargs.get('debug'):
                print(f"{text}: {sentiment}")
            return 2
        else:  # negative
            sentiment = 1
            if self.kwargs.get('debug'):
                print(f"{text}: {sentiment}")
            return 1
            # print({'text': tweet['text'], 'place': tweet['place']['name'], 'sentiment': self.pred[sentiment]})

    def naive_bayes(self, array=None):
        if array is not None:
            path = os.path.join(self.kwargs.get('root_path'), 'analysis', 'nb_model.pkl')
            with open(path, 'rb') as f:
                cls = pickle.load(f)

            self.pred = cls.predict(array)
            # self.pred = cls.predict_proba(array)
            # print(self.pred)
            # with open('pred_results.txt', 'w') as f:
            #     [f.write(str(p)+'\n') for p in pred]
        else:
            Y = self.combined_array[:,-1]
            X = self.combined_array[:,:-1]
            x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=155)
            clf = MultinomialNB(alpha=1)
            clf.fit(x_train, y_train)
            self.pred = clf.predict(x_test)
            # self.pred = clf.predict_proba(x_test)
            print(classification_report(y_test, self.pred, target_names=['pos','neg']))

            path = os.path.join(self.kwargs.get('root_path'), 'analysis', 'nb_model.pkl')
            with open(path, 'wb') as f:
                print("***** writing model *****")
                pickle.dump(clf, f)

    def svm(self):
        """ runs but is too slow """
        Y = self.combined_array[:, -1]
        X = self.combined_array[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=155)

        # build pipeline
        svc = SVC()
        pipe = Pipeline([('svc', svc)])
        # set params
        svm_paramaters = [{'svc__C': [1.], 'svc__kernel': ['linear'], 'svc__gamma': [1.]}]
        # fit data to gridsearch model
        start = timeit.default_timer()
        grid_search_svm = GridSearchCV(pipe, svm_paramaters, cv=2, n_jobs=-1)
        grid_search_svm.fit(x_train, y_train)
        print(f'Runtime for SVM:{timeit.default_timer() - start}')
        print(grid_search_svm.best_params_)
        pred = grid_search_svm.predict(x_test)
        print(classification_report(y_test, pred, target_names=['pos', 'neg']))
        return grid_search_svm

    def write_json(self, f_name=None):
        """ write tweet to json file """
        print("\n***** writing json *****")
        city_rating = defaultdict(lambda: defaultdict(lambda: 0))

        # use for debugging classification results
        i = 0
        if self.kwargs.get('debug') and not self.kwargs.get('vader'):
            for tweet in self.tweets:
                print(f"{tweet['text']}: {self.pred[i]}")
                i+=1

        # build city rating dict
        j = 0
        if self.kwargs.get('vader', False):
            for tweet in self.tweets:
                if tweet.get('place') is not None:
                    t = {'text': tweet['text'], 'place': tweet['place']['full_name'],
                         'sentiment': '{}'.format(tweet['sentiment'])}
                    name = tweet['place']['name']
                    if name not in city_rating.keys() and int(tweet['sentiment']) == 0:
                        city_rating[name]['pos'] = 1
                    elif name not in city_rating.keys() and int(tweet['sentiment']) == 1:
                        city_rating[name]['neg'] = 1
                    elif name not in city_rating.keys() and int(tweet['sentiment']) == 2:
                        city_rating[name]['neu'] = 1
                    elif name in city_rating.keys() and int(tweet['sentiment']) == 0:
                        city_rating[name]['pos'] += 1
                    elif name in city_rating.keys() and int(tweet['sentiment']) == 1:
                        city_rating[name]['neg'] += 1
                    else:
                        city_rating[name]['neu'] += 1

        else:
            for tweet in self.tweets:
                if tweet.get('place') is not None:
                    try:
                        t = {'text': tweet['text'], 'place': tweet['place']['full_name'],
                             'sentiment': '{}'.format(self.pred[j])}
                        name = tweet['place']['name']
                        if name not in city_rating.keys() and int(self.pred[j]) == 0:
                            city_rating[name]['pos'] = 1
                        elif name not in city_rating.keys() and int(self.pred[j]) == 1:
                            city_rating[name]['neg'] = 1
                        elif name in city_rating.keys() and int(self.pred[j]) == 0:
                            city_rating[name]['pos'] += 1
                        else:
                            city_rating[name]['neg'] += 1
                    except Exception:
                        print(traceback.format_exc())
                else:
                    j+=1
                    continue
                j+=1

        # write city rating to file
        # with open(os.path.join(self.kwargs.get('root_path'), 'output', f_name), 'w') as f:
        with open(f_name, 'w') as f:
            f.write(json.dumps(city_rating) + '\n')
            # write tweets to same file
            j = 0

            # write sentiment results for naive bayes classification
            if not self.kwargs.get('vader'):
                for tweet in self.tweets:
                    if tweet.get('place'):
                        d = {'text': tweet['text'], 'place': tweet['place']['name'], 'sentiment': self.pred[j]}
                        # if self.kwargs.get('debug'):
                        #     print('{}: {}'.format(d['text'],d[self.pred[j]]))
                        f.write(json.dumps(d) + '\n')
                        j += 1
            # write sentiment results for vader
            else:
                print('writing vader results')
                for tweet in self.tweets:
                    d = {'text': tweet['text'], 'place': tweet['place']['name'], 'sentiment': tweet['sentiment']}
                    f.write(json.dumps(d) + '\n')

def main():
    ROOT_PATH = os.path.realpath('..')
    TWEET_FILE_PATH = os.path.join(ROOT_PATH, 'tweet_dir')

    # train model
    # t = SA(train=True, root_path=ROOT_PATH, stopwords=False, debug=False)

    # predict tweets using Naive Bayes Model
    t = SA(tweet_files_path=TWEET_FILE_PATH, root_path=ROOT_PATH, train=False, debug=True, stopwords=False)

    # predict tweets using Vader for debugging only
    # t = SA(root_path=ROOT_PATH, vader=True, debug=True, tweet_files_path=TWEET_FILE_PATH)

    # predict tweets using Vader no debugging
    # t = SA(root_path=ROOT_PATH, vader=True, debug=False, tweet_files_path=TWEET_FILE_PATH)


if __name__ == "__main__":
    main()
