#!/usr/bin/env python

from searchtweets import ResultStream, gen_rule_payload, load_credentials
from searchtweets import collect_results
from searchtweets.utils import write_result_stream
from searchtweets.utils import write_ndjson
import json


class TwitterPremiumQuery:
    def __init__(self, **kwargs):
        """ Leverage the twitter provided searchtweets package as a wrapper to interact with the premium search endpoint
        Args:
            cred_file(str): location of credentials
            do_sandbox(boolean): use sandbox development environment, default is True
            rules: search key:value pairs to filter search tweet
            premium_search_args(func): authenticate using the searchtweets package api
        """
        # class variables
        self.kwargs = {k:v for k,v in kwargs.items()}
        self.cred_file = self.kwargs.get('cred_file', 'twitter_keys.yaml')
        self.from_date = self.kwargs.get('from_date', '201903120000')
        self.to_date = self.kwargs.get('to_date', '201904050000')
        self.do_sandbox = self.kwargs.get('do_sandbox', True)
        self.do_count = self.kwargs.get('do_count', False)
        self.premium_search_args = None
        self.raw_rule = None

        # method calls
        self.authenticate()
        self.create_search_rules()
        if self.do_count:
            self.get_rule_count()
        else:
            self.create_search_payload()
            # self.search()
            # self.write_stream()

    def authenticate(self):
        """ authenticate using the searchtweets api with yaml configs from twitter_keys.yaml """
        if self.do_sandbox:
            self.premium_search_args = load_credentials(filename=self.cred_file,
                                                        yaml_key='full_tweets_api_sandbox', env_overwrite=False)
        else:
            self.premium_search_args = load_credentials(filename=self.cred_file,
                                                        yaml_key='search_tweets_api', env_overwrite=False)

    def create_search_rules(self):
        """ create search api rules using gen_rule_payload

        # https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
        # the following operators only work with premium--not sandbox
            -is:retweet has:geo has:profile_geo profile_region profile_country
        """
        # TODO: get lat/lon for all states: https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population

        # All US
        self.raw_rule = """
        -is:retweet
        (cpt marvel OR (captain marvel))
        (profile_country:US)
        """

        # Seattle, WA rule
        self.raw_rule = """
        -is:retweet
        (cpt marvel OR (captain marvel))
        (place_country:US bounding_box:[-122.436232 47.495315 -122.2249728 47.734319] OR place:"Seattle, WA")
        """

        # # Manhattan, NY rule
        # self.raw_rule = """
        # (cpt marvel OR (captain marvel))
        # (bounding_box:[-74.026675 40.683935 -73.910408 40.877483] OR place:"Manhattan, NY")
        # """

        # # Atlanta, GA rule
        # self.raw_rule = """
        # (cpt marvel OR (captain marvel))
        # (bounding_box:[-84.576827 33.6475029 -84.289385 33.8868859] OR place:"Atlanta, GA")
        # """

        # # Los Angeles, CA rule
        # self.raw_rule = """
        # (cpt marvel OR (captain marvel))
        # (point_radius:[-118.249805 34.046803 25mi] OR place:"Los Angeles, CA")
        # """

        # Chicago, IL rule
        # self.raw_rule = """
        # -is:retweet
        # (cpt marvel OR (captain marvel))
        # (point_radius:[-87.668568 41.828946 25mi] OR place:"Chicago, IL")
        # """

    def create_search_payload(self):
        if self.do_sandbox:
            self.rule = gen_rule_payload(self.raw_rule, results_per_call=100,
                                         from_date=self.from_date, to_date=self.to_date)

        else:
            self.rule = gen_rule_payload(self.raw_rule, results_per_call=500,
                                         from_date=self.from_date, to_date=self.to_date)

    def get_rule_count(self):
        """ before calling the production api, get a count of the tweets that match the rule """
        rule_count = gen_rule_payload(self.raw_rule,
                                      from_date=self.from_date, to_date=self.to_date,
                                      results_per_call=500, count_bucket='day')

        counts_list = collect_results(rule_count, max_results=500, result_stream_args=self.premium_search_args)
        [print(count) for count in counts_list]

    def search(self):
        self.tweets = collect_results(self.rule, max_results=100, result_stream_args=self.premium_search_args)
        [print(tweet.all_text, end='\n\n') for tweet in self.tweets[0:100]]

    def write(self):
        with open('atlanta_mar09_apr05.json', 'w') as f:
            f.write(json.dumps(self.tweets))

    def write_stream(self):
        """ write ResultStream object to disk using the write_ndjson utility """
        stream = ResultStream(**self.premium_search_args, rule_payload=self.rule, max_results=62000)
        columns = []
        for _ in write_ndjson('US_apr02_apr09_some.json', stream.stream()):  # exhaust generator
            pass


def main():
    # t = TwitterPremiumQuery(do_sandbox=True, from_date='201903090000', to_date='201903100000')
    t = TwitterPremiumQuery(do_sandbox=False, from_date='201904020000', to_date='201904090000')
    # t.search()
    # t.write()
    t.write_stream()


if __name__ == "__main__":
    main()
