#!/usr/bin/env python3
"""
  setup.py file
"""
import os.path
import re
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

PACKAGES = find_packages(HERE)

with open(os.path.join(HERE, 'Readme.md')) as readme_file:
  README = readme_file.read()

with open(os.path.join(HERE, 'ccsonggenerator', '__init__.py')) as init_file:
  METADATA = dict(re.findall(r'__([a-z]+)__ = "([^"]+)', init_file.read()))

setup(
  name="ccsonggenerator",
  description=(
    "A tool for auto generating unique song lyrics from a dataset of "
    "traditional Irish folk music."
  ),
  long_description=README,
  version=METADATA['version'],
  author="Ryan Jennings",
  author_email="ryan.jennings1@ucdconnect.ie",
  url="https://github.com/RyanJennings1/CC-Song-Generator",
  license="FIXME",
  packages=PACKAGES,
  scripts=[
    'bin/ccsonggenerator',
  ],
  install_requires=[
    'tensorflow==1.13.1',
    'tweepy',
    'pyenchant',
    'nltk'
  ],
  classifiers=[
    'Development Status :: 1 - Alpha',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development :: Libraries :: Application Frameworks',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ]
)
