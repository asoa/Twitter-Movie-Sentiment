#!/usr/bin/env python

from util import helpers
import os
from analysis.sentiment_analysis import SA

def main():
    ROOT_PATH = os.getcwd()
    TWEET_FILE_PATH = os.path.join(ROOT_PATH, 'tweet_dir')

    # train model
    # t = SA(train=True, root_path=ROOT_PATH, stopwords=False, debug=False)

    # predict tweets using Naive Bayes Model
    # t = SA(tweet_files_path=TWEET_FILE_PATH, root_path=ROOT_PATH, train=False, debug=True, stopwords=False)

    # predict tweets using Vader for debugging only
    # t = SA(root_path=ROOT_PATH, vader=True, debug=True, tweet_files_path=TWEET_FILE_PATH)

    # predict tweets using Vader no debugging
    # t = SA(root_path=ROOT_PATH, vader=True, debug=False, tweet_files_path=TWEET_FILE_PATH)


if __name__ == "__main__":
    main()
