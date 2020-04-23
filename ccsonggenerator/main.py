"""
  name: main.py
  author: Ryan Jennings
  date: 2020-04-21
"""
import operator
import os

from typing import Dict, Tuple

import ccsonggenerator.txtutil as txt

from ccsonggenerator.song_generator import SongGenerator
from ccsonggenerator.twitter import Twitter

def main(args: Dict[str, bool]) -> None:
  """
  main method

  To do:
  - More validation?
  - Get to run on computer every hour
  - Flesh out Readme more
  """
  song_generator = SongGenerator()
  consumer_key, consumer_secret, access_token, access_token_secret = load_env_vars()
  twitter_api = Twitter(consumer_key,
                        consumer_secret,
                        access_token,
                        access_token_secret)

  if args.train:
    song_generator.train()
  if args.run:
    # Generate multiple and pick most logical
    paragraph = best_generated_paragraph(10, song_generator)
    print("\n\n\n\n")
    print(paragraph)
    #twitter_api.tweet_generated_text(song_generator.get_output_filename())
  if args.analyse:
    # pull down tweet data
    twitter_api.retrieve_existing_tweet_data(write_to_file=True)

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

def best_generated_paragraph(num_to_generate, generator: SongGenerator) -> str:
  """
  Return the paragraph with the hightest percentage of real words
  """
  validities: Dict[str, int] = {}
  for _ in range(num_to_generate):
    paragraph: str = "\n".join(generator.generate().split("\n")[:-1])
    validities[paragraph] = txt.paragraph_validity(paragraph)
  sorted_validities = reversed(sorted(validities.items(), key=operator.itemgetter(1)))
  return next(sorted_validities)[0]