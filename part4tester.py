import better_objects as b
import faq_config
import random
import sys
import lesk
import nlp_config
from nltk.parse.stanford import StanfordDependencyParser

#TODO: this should be in a central place
dependency_parser = StanfordDependencyParser(path_to_jar=nlp_config.path_to_stanford_jar, path_to_models_jar=nlp_config.path_to_stanford_models_jar)

class Annealer(object):
  #all of the parameters are functions. e = energy, lower is better. p = probability.
  def __init__(self, neighbor, e, p, temperature):
    self.neighbor = neighbor
    self.e = e
    self.p = p
    self.temperature = temperature

max_temp = 100    

def neighbor(s, adj_min = 0.001, adj_max = 0.005):
  #random index, random minor adjustment
  idx = random.randrange(len(s.weights))
  adj = random.uniform(adj_min, adj_max)
  if random.randrange(2) == 0:
    adj *= -1
  new_weights = [w for w in s.weights]
  new_weights[idx] += adj
  return new_weights

#larger e is bad
#larger change in e is bad
#smaller temperature is bad (in terms of accepting the change)
def probability(e, e_prime, temperature):
  if e_prime < e:
    return 1
  return (1 - (e_prime - e)) * temperature / max_temp
  
#between 0 and max_temp
def temperature(time): #time is k / kmax
  return (1 - time) * max_temp

#needs to be between 0 and 1
def energy(state, weights):
  scores = state.get_scores(weights)
  
  #slot = 0
  #x = sorted([(val, key) for key, val in scores.items()], reverse=True)
  
  #for xx in x:
  #  if xx[1] == state.best_answer:
  #    break
  #  slot += 1
  
  #super simple energy function. invert the score
  #this is probably too naive, but we will try it.
  return (1 - scores[state.best_answer])# * slot / len(x)

class State(object):
  def __init__(self, q_features, as_features, weights, best_answer):
    self.weights = weights #never used
    self.best_answer = best_answer
    self.score_vectors = []
    for ix, af in enumerate(as_features):
      score_vector = [get_score_simple(q_features.tokens, af.tokens),
                      get_score_simple(q_features.tokens_no_stops, af.tokens_no_stops),
                      get_score_simple(q_features.lemmas, af.lemmas),
                      get_score_simple(q_features.stems, af.stems),
                      get_score_simple(q_features.pos_tags, af.pos_tags),
                      get_score_simple(q_features.synset_lemmas, af.synset_lemmas),
                      get_score_simple(q_features.antonym_lemmas, af.antonym_lemmas),
                      get_score_simple(q_features.hyponym_lemmas, af.hyponym_lemmas),
                      get_score_simple(q_features.hypernym_lemmas, af.hypernym_lemmas),
                      get_score_simple(q_features.part_meronym_lemmas, af.part_meronym_lemmas),
                      get_score_simple(q_features.part_holonym_lemmas, af.part_holonym_lemmas),
                      get_score_simple(q_features.member_meronym_lemmas, af.member_meronym_lemmas),
                      get_score_simple(q_features.member_holonym_lemmas, af.member_holonym_lemmas),
                      get_score_simple(q_features.substance_meronym_lemmas, af.substance_meronym_lemmas),
                      get_score_simple(q_features.substance_holonym_lemmas, af.substance_holonym_lemmas),
                      get_score_simple(q_features.wn_definitions, af.wn_definitions),
                      get_score_simple(q_features.wn_examples, af.wn_examples),
                      get_score_simple(q_features.depgraph_deps, af.depgraph_deps),
                      get_score_simple(q_features.depgraph_rels, af.depgraph_rels)]
      self.score_vectors.append(score_vector)
      
  def get_scores(self, used_weights = None):
    scores = dict()
    for ix, sv in enumerate(self.score_vectors):
      effective_score = b.score_features(sv, used_weights)
      scores[ix + 1] = effective_score
    return scores
                      
def lt_default(a, b): return a < b

def get_score_simple(arr1, arr2):
  if len(arr1) == 0 or len(arr2) == 0:
    return 0
  math_vecs = b.get_math_vectors(arr1, arr2, lt_default)
  return b.cosine_similarity(math_vecs[0], math_vecs[1])

faqs = faq_config.getFAQs()
question = "At what speed do hummingbirds fly in the air?"#"What do hummingbirds eat?"#"What is the lifecycle of a hummingbird like as it grows from birth as a child to death?"#"Describe the hummingbird's lifecycle."#"What do hummingbirds eat?"#"At what speed do hummingbirds fly in the air?"

as_features = b.get_answers_features(faqs)
q_features = b.TextFeatureExtraction(question)

#this should set up the synsets
b.load_all_synsets(as_features)
q_features.synsets = lesk.get_synsets_from_features(q_features)

#this should set up dependency graphs
b.load_all_depgraphs(as_features)
q_features.depgraphs = [dg for dg in dependency_parser.raw_parse(question)] #TODO: not the best way to do this. also iter to make an array

for f in as_features:
  f.add_wordnet_features()
  f.add_depgraph_features()
q_features.add_wordnet_features()
q_features.add_depgraph_features()

feature_count = 19 #TODO: hardcoded
learned_weights = [1] * feature_count

max_steps = 25000
state = State(q_features, as_features, learned_weights, 5)#11)#10)#11) #5)
anneal = Annealer(neighbor, energy, probability, temperature)
for k in range(max_steps):
  t = anneal.temperature(k / max_steps)
  new_weights = anneal.neighbor(state)
  e_old = anneal.e(state, state.weights)
  e_new = anneal.e(state, new_weights)
  if anneal.p(e_old, e_new, t) >= random.random():
    state.weights = new_weights
    
  if k % 20 == 0:
    print("k: %5d, last energy: %f. weights = %s" % (k, e_old, state.weights)) #TODO: might not be e_old

learned_weights = state.weights
dict_scores = sorted([(ascore, qnum) for qnum, ascore in state.get_scores(learned_weights).items()], reverse=True)

for pair in dict_scores:
  print("%2d: %f" % (pair[1], pair[0]))