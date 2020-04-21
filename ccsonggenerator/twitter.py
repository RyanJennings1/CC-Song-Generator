#!/usr/bin/env python
"""
  name: twitter.py
  author: Ryan Jennings
  date: 2020-04-21
"""
from typing import Any

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
