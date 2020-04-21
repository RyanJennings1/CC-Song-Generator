#!/usr/bin/env python
"""
  name: twitter.py
"""
import os

from typing import Tuple

import tweepy

def load_env_vars() -> Tuple[str, str, str, str]:
  """
  Return twitter credentials from environment

  return: Tuple[str, str, str, str]
  """
  consumer_key = os.environ.get('twitter_api_key')
  consumer_key_secret = os.environ.get('twitter_api_key_secret')
  access_token = os.environ.get('twitter_access_token')
  access_token_secret = os.environ.get('twitter_access_token_secret')
  if not consumer_key or not consumer_key_secret or not access_token or not access_token_secret:
    raise Exception('Invalid api keys')
  return consumer_key, consumer_key_secret, access_token, access_token_secret

def post_tweet(api, text):
  """
  post a tweet
  """
  api.update_status(text)

def main():
  """
  main method
  """
  consumer_key, consumer_secret, access_token, access_token_secret = load_env_vars()
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth)

  post_tweet(api, 'test')

if __name__ == '__main__':
  main()
