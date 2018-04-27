import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('stopwords')

stops = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def remove_lemma_and_stops(lemmas, lemma):
  return {x for x in lemmas if x != lemma and x not in stops and x is not None}

def compute_overlap(synset, lemma_set, lemma):
  def get_lemma_set(text):
    #tokenize, lemmatize, and remove stops
    ret = [lemmatizer.lemmatize(x) for x in nltk.word_tokenize(text)]
    return remove_lemma_and_stops(ret, lemma)
    
  def get_lemma_set_from_synset():
    ret = get_lemma_set(synset.definition())
    for example in synset.examples():
      ret |= get_lemma_set(example)
    return ret
    
  synsets_lemmas = get_lemma_set_from_synset()
  return len(synsets_lemmas & lemma_set)
  
#get the synset that most closely matches a lemma
#need to modify to allow restriction to a specific part of speech
def get_lemma_synset(lemma, lemma_neighbors):
  lemma_set = remove_lemma_and_stops(lemma_neighbors, lemma)
  
  synsets = wn.synsets(lemma)
  
  if not synsets:
    return None
    
  best_option = (synsets[0], 0) #best synset, best overlap
  
  for candidate_syn in synsets:
    overlap_value = compute_overlap(candidate_syn, lemma_set, lemma)
    
    if overlap_value > best_option[1]:
      best_option = (candidate_syn, overlap_value)
      
  return best_option[0]
  
#takes a textfeatureextraction object
def get_synsets_from_features(tfe):
  return [get_lemma_synset(lemma, tfe.lemmas) for lemma in tfe.lemmas]