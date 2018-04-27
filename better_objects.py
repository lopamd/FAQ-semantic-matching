import nltk
import base_objects
import nltk_objects
from collections import Counter
import sklearn.metrics
import numpy as np
from nltk.corpus import wordnet as wn

synset_folder = 'data/synsets/'
synset_filename_format = "%d_%s.txt" #%d is from 1 to 50, %s is question or answer

#TODO: normalize words??
#TODO: do lemmatize and stem need context?? tokens were already sorted

stops = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

#array of array of sorted answer tokens.
def do_tokenize(text):
  return sorted(nltk.word_tokenize(text))
    
#array of array of sorted answer tokens, not including stop words.
def do_tokenize_no_stops(tokens):
  return [w for w in tokens if w not in stops]
    
#array of array of sorted lemmas including stop words
def do_lemmatize(tokens):
  return [lemmatizer.lemmatize(w) for w in tokens]
  
#array of array of sorted stems including stop words
def do_stem(tokens):
  return [stemmer.stem(w) for w in tokens]
  
#array of array of tuples of the form ('word', 'pos')
def do_pos_tag(text):
  return sorted(nltk.pos_tag(nltk.word_tokenize(text)))
  
class TextFeatureExtraction(object):
  def __init__(self, text):
    self.tokens = do_tokenize(text)
    self.tokens_no_stops = do_tokenize_no_stops(self.tokens)
    self.lemmas = do_lemmatize(self.tokens)
    self.stems = do_stem(self.tokens)
    self.pos_tags = do_pos_tag(text)
    self.synsets = []
    
#takes a text feature extraction and a filename and hooks you up with the synsets
def add_synsets(tfe, filename):
  lines = [line.rstrip('\n') for line in open(filename)]
  synset_names = [line.split()[1] for line in lines] #grab the synset names
  tfe.synsets.extend([wn.synset(synset_name) for synset_name in synset_names])
  
def load_all_synsets(tfes):
  current = 1
  for tfe in tfes:
    filename_question = synset_folder + (synset_filename_format % (current, "question"))
    filename_answer = synset_folder + (synset_filename_format % (current, "answer"))
    add_synsets(tfe, filename_question)
    add_synsets(tfe, filename_answer)
    current += 1
    
def get_answers_features(qapairs):
  ret = []
  for qa in qapairs:
    ret.append(TextFeatureExtraction(qa.answer))
  return ret
  
def get_math_vectors(items_one, items_two, lt):
  counters = (Counter(items_one), Counter(items_two))
  
  #sort because we're going to be walking the lists
  items = (sorted(counters[0].items()), sorted(counters[1].items()))
  
  vectors = ([], [])
  
  key_indices = (0, 0)
  
  while key_indices[0] < len(items[0]) and key_indices[1] < len(items[1]):
    itempair = (items[0][key_indices[0]], items[1][key_indices[1]])
    if lt(itempair[0][0], itempair[1][0]): #comparing the keys
      vectors[0].append(itempair[0][1]) #add the count to the math vector
      vectors[1].append(0)
      key_indices = (key_indices[0] + 1, key_indices[1])
    elif lt(itempair[1][0], itempair[0][0]):
      vectors[0].append(0)
      vectors[1].append(itempair[1][1]) #add the count to the math vector
      key_indices = (key_indices[0], key_indices[1] + 1)
    else:
      vectors[0].append(itempair[0][1]) #add the count to the math vector
      vectors[1].append(itempair[1][1]) #add the count to the math vector
      key_indices = (key_indices[0], key_indices[1] + 1)

  while key_indices[0] < len(items[0]):
    vectors[0].append(items[0][key_indices[0]][1]) #add the count to the math vector
    vectors[1].append(0)
    key_indices = (key_indices[0] + 1, key_indices[1])

  while key_indices[1] < len(items[1]):
    vectors[0].append(0)
    vectors[1].append(items[1][key_indices[1]][1]) #add the count to the math vector
    key_indices = (key_indices[0], key_indices[1] + 1)

  return vectors
    
def cosine_similarity(a, b):
  #the semantics of cosine_similarity are annoying.
  #it must make sense in general because it's really annoying.
  return sklearn.metrics.pairwise.cosine_similarity([a], [b])[0][0] #seriously, it's a number in a nested array
  
#a and b are already scored vectors
def score_features(scores, weights):
  weighted_sims = [c * d for c, d in zip(scores, weights)]
  return np.linalg.norm(weighted_sims) / np.linalg.norm(weights)