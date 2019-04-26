#!/usr/bin/env python

import twitter
from twitter import TwitterStream
from streaming.authenticate import Authenticate
import pymongo
from pymongo import MongoClient
import traceback
import json


class Stream:
    def __init__(self, **kwargs):
        self.kwargs = {k:v for k,v in kwargs.items()}
        self.q = self.kwargs.get('query', 'shazam,shazam!')
        self.db = None
        self.coll = None
        self.coll_name = self.kwargs.get('coll_name')
        self.db_name = self.kwargs.get('db_name')
        self.twitter_api = self.kwargs.get('twitter_api', None)
        self.db_client = MongoClient(host='mongo_db', port=27017)  # connect to db

    def db_init(self):
        """ Create database and document collection """
        try:
            # self.db = self.db_client.tweets  # create db
            # self.coll = self.db.shazam  # create document collection
            self.db = self.db_client[self.db_name]
            self.coll = self.db[self.coll_name]
        except Exception as e:
            print(e)

    def start_stream(self):
        """ creates the connection to the twitter streaming endpoint"""
        _stream = TwitterStream(auth=self.twitter_api)
        # locations[seattle, new york]
        tweet_iterator = _stream.statuses.filter(track=self.q, language='en',
                             locations='-122.436232,47.495315,-122.2249728,47.734319,-74.255641,40.495865,-73.699793,40.91533')

        for tweet in tweet_iterator:
            print(tweet['text'], end='\n\n')
            self.write_db(tweet)

    def write_db(self, tweet):
        """ write to mongo db """
        self.coll.insert_one(tweet)

    def parse_args(self):
        """ parse cmd line args """
        #TODO


def main():
    api = Authenticate(creds_file='twitter_creds.BU')
    try:
        twitter_stream = Stream(twitter_api=api.auth, query='shazam,shazam!')
        twitter_stream.db_init()
        twitter_stream.start_stream()

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    main()
