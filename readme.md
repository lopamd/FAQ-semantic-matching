

## Semantic Matching FAQ Application

### Problem Description

Given 50 frequently asked questions (FAQs) about hummingbirds, the task is to implement a naïve approach and a more sophisticated approach using Natural Language Processing techniques to retrieve the most related FAQs when presented with a user’s question about hummingbirds. For instance, when the user asks “How many years will a hummingbird live?”, the system must attempt to provide a ranked list of the most relevant FAQs. The naïve approach should treat the questions and answers as bags of words and match the word tokens against a user’s question. The other approach must use tokens, lemmas, stems, parts of speech, parse trees, and Wordnet to find matches in a more intelligent way.


### Various modules in solution:

Collection of features
Processing of features
Calculation of Weights
Learning weights
Final score calculation
Ranking of answers

A details description of each module can be found in [final report]( project_report.pdf)

### Programming Tools:
* Python 3

	Python 3 is the primary programming language used for this project.
* Java

	While no Java code was written by the team members, jar libraries from the Stanford Parser were used for dependency parsing.
* NLTK

	Natural Language Toolkit. This is a library for Python that provides a host of natural language processing tools. These include access to wordnet, various corpora, and dependency parsers.
* Stanford Parser

	Stanford’s dependency parser was used to get dependency trees from questions and answers. We used a python library that wrapped the java libraries.
* Wordnet

	The wordnet corpus was used via NLTK. We collected synsets via the Lesk algorithm and used their similarities and definitions and examples. We also used the Brown corpus for information content that was exposed via Wordnet.
* Brown Corpus

	We used the Brown corpus that was provided with Wordnet for the information content when we took the JCN similarity of various synsets.
* Numpy

	Numpy is a Python library that we used for linear algebra operations such as taking the norm of a vector.
* Sklearn

	Sklearn is a Python library that we used for cosine similarity calculations.
* Brown Corpus

	We used the Brown corpus that was provided with Wordnet for the information content when we took the JCN similarity of various synsets.

### Output:
Please refer [final report]( project_report.pdf).
