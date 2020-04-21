"""
  name: main.py
  author: Ryan Jennings
  date: 2020-04-21
"""
import os

from typing import Dict, Tuple

from ccsonggenerator.song_generator import SongGenerator
from ccsonggenerator.twitter import Twitter

def main(args: Dict[str, bool]) -> None:
  """
  main method
  """
  song_generator = SongGenerator()
  if args.train:
    song_generator.train()
  elif args.run:
    song_generator.generate()
    # Generate multiple and pick most logical
    consumer_key, consumer_secret, access_token, access_token_secret = load_env_vars()
    twitter_api = Twitter(consumer_key,
                          consumer_secret,
                          access_token,
                          access_token_secret)
    twitter_api.tweet_generated_text(song_generator.get_output_filename())

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
