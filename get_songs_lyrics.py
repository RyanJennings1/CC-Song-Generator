#!/usr/bin/env python
"""
  name: get_songs_lyrics.py
  author: Ryan Jennings
  date: 2020-04-06
"""

def main():
  """
  main method
  """
  with open('song-titles.txt', 'r') as song_file:
    for line in song_file:
      with open(f'lyrics/{line.rstrip()}.txt', 'w') as lyric_file:
        pass

if __name__ == '__main__':
  main()
