The following dependencies need to be installed:

  pip install nltk

* If you are on python2, you need to switch to python 3.
by doing this on mac:
brew install python3
pip3 install nltk

You need to use python3 to execute any python script.
This is convient if you do not want to change your default python enviornment to python 3.
  
If nltk gives an error about 'charmap' encoding or decoding not working and you're on Windows, run the following command in your terminal:

  set PYTHONIOENCODING="UTF-8"
  
Java is needed for the Stanford Parser, used for dependency parsing. At least version 1.8 is needed. Use a similar command to this on Windows to set JAVAHOME:

  set JAVAHOME=C:\Program Files\Java\jre-10
 
* If you are on Mac:

1. Install Java using
brew cask install java
if homebrew is not configured 
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew cask install java
 
The Stanford files are not included in this release. They need to be manually downloaded from https://stanfordnlp.github.io/CoreNLP/#programming-languages-and-operating-systems and added to the root of the repository. Look in nlp_config.py for hints about where to place the downloaded artifacts.