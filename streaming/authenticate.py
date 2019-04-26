#!/usr/bin/env python3

import twitter
import os
import ast


class Authenticate(object):
    """ Authenticates to the twitter API

    Attributes:
        # add keys to twitter_creds.txt for script to read from
        consumer_key: key to identify the client
        consumer_secret: client password used to authenticate with twitter oauth
        oauth_token: key to define privileges
        oauth_secret: key used with token as password
    """

    def __init__(self, *args, **kwargs):
        self.kwargs = {k: v for k, v in kwargs.items()}
        self.consumer_key = self.kwargs.get('consumer_key')
        self.consumer_secret = self.kwargs.get('consumer_secret')
        self.oauth_token = self.kwargs.get('oauth_token')
        self.oauth_secret = self.kwargs.get('oauth_secret')
        self.auth = None
        self.twitter_api = None
        self.creds_file = self.kwargs.get('creds_file', None)
        self.parse_file_creds()
        self.twitter_authenticate()

    def parse_file_creds(self):
        with open(self.creds_file, 'r') as f:
            creds = ast.literal_eval(f.read())
            self.consumer_key = creds.get('CONSUMER_KEY')
            self.consumer_secret = creds.get('CONSUMER_SECRET')
            self.oauth_token = creds.get('OAUTH_TOKEN')
            self.oauth_secret = creds.get('OAUTH_TOKEN_SECRET')
            self.auth = twitter.oauth.OAuth(self.oauth_token, self.oauth_secret, self.consumer_key,
                                            self.consumer_secret)

    def twitter_authenticate(self):
        self.twitter_api = twitter.Twitter(auth=self.auth)


def main():
    a = Authenticate(creds_file='twitter_creds.BU')
    print(a.twitter_api)


if __name__ == "__main__":
    main()
