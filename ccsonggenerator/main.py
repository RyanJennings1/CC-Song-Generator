"""
  name: main.py
  author: Ryan Jennings
  date: 2020-04-21
"""

from ccsonggenerator.song_generator import SongGenerator

def main():
  """
  main method
  """
  song_generator = SongGenerator()
  song_generator.generate()
