#!/usr/bin/env python
"""
  name: twitter.py
  author: Ryan Jennings
  date: 2020-04-21
"""
import csv

from typing import Any, Dict, List

import tweepy

class Twitter():
  """
  Twitter API interface class
  """
  def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
    """
    initialise values
    """
    self.consumer_key = consumer_key
    self.consumer_secret = consumer_secret
    self.access_token = access_token
    self.access_token_secret = access_token_secret
    auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
    auth.set_access_token(self.access_token, self.access_token_secret)
    self.connection = tweepy.API(auth)

  def get_connection(self) -> Any:
    """
    return the connection object - type tweepy.API
    """
    return self.connection

  def tweet_generated_text(self, text) -> None:
    """
    Publish a tweet of text
    """
    self.connection.update_status(text)

  def retrieve_existing_tweet_data(self, write_to_file=False) -> List[Dict[str, Any]]:
    """
    Retrieve the data on the existing tweets
    """
    tweet_data: List[Dict[str, Any]] = []
    responses = self.connection.home_timeline()
    for tweet in responses:
      if tweet.favorite_count > 0 or tweet.retweet_count > 0:
        tweet_data.append({
          'id': tweet.id,
          'fav_count': tweet.favorite_count,
          'retweet_count': tweet.retweet_count,
          'text': tweet.text
        })
    tweets = sorted(tweet_data, key=lambda x: x['fav_count'])
    if write_to_file and len(tweets) > 0:
      with open('tweet_statistics.csv', 'w') as tw_st_file:
        csv_writer = csv.DictWriter(tw_st_file, delimiter=',', fieldnames=list(tweets[0].keys()))
        csv_writer.writeheader()
        for tweet in tweets:
          csv_writer.writerow(tweet)
    return tweets
