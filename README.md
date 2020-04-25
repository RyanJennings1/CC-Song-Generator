# Introduction
This is the repo for **Auto Song Generation** of Irish Folk songs.  
This is part of my COMP47410 Computational Creativity Assignment

# :ledger: Index
- [Introduction](#introduction)
- [:ledger: Index](#ledger-index)
- [:beginner: About](#beginner-about)
- [:electric_plug: Installation](#electricplug-installation)
- [:sunny: Usage](#sunny-usage)
- [:wrench: Development](#wrench-development)
- [:star2: Credit/Acknowledgment](#star2-creditacknowledgment)

# :beginner: About
This project is part of the final assignment for COMP47410 Computational Creativity. The repo contains a machine learning model that takes a folder of lyrics and outputs generated text before posting it on twitter. The lyrics directory can be swapped with any text type data to train then generated.

#  :electric_plug: Installation
```bash
$ git clone https://github.com/RyanJennings1/CC-Song-Generator.git
```

Install the project before use to include all the dependencies:
- tensorflow==1.13.1
- tweepy
- pyenchant
- nltk

```bash
$ python3 setup.py install
```  

**Also Required**
A Twitter developer account is needed to get Twitter API secrets which can be used to make the bot post to twitter.

Logins for making a connection to Twitter are brought in from user's shell environment values `twitter_api_key`, `twitter_api_key_secret`, `twitter_access_token` and `twitter_access_token_secret`

# :sunny: Usage
After installation and being built just run the binary
```bash
$ ./bin/ccsonggenerator --help
usage: ccsonggenerator [-h] [--train] [--run] [--analyse] [--version]

optional arguments:
  -h, --help     show this help message and exit
  --train, -t    Train the model on lyrics data
  --run, -r      Run the model to generate an output
  --analyse, -a  Analyse existing tweets published
  --version, -v  Print version
```

#  :wrench: Development
If anybody wants to contribute to this project or fork it feel free.

# :star2: Credit/Acknowledgment
[Ryan Jennings](ryan.jennings1@ucdconnect.ie)  
[burliEnterprises](https://github.com/burliEnterprises/tensorflow-shakespeare-poem-generator)