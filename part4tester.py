import better_objects as b
import faq_config
import random
import sys
import lesk
import nlp_config
import base_objects
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import wordnet_ic
import numpy as np

brown_ic = wordnet_ic.ic('ic-brown.dat')

test_questions = [
  "When is hummingbird season?",
  "Where do hummingbirds go in the winter?",
  "Where do hummingbirds live?",
  "What is the reproduction process like for hummingbirds?",
  "How fast does a hummingbird fly?",
  "Do hummingbirds damage flowers?",
  "Do hummingbirds reuse their nest?",
  "How much nectar does a hummingbird consume in a day?",
  "Do hummingbirds eat termites?",
  "What is a hummingbird's lifecycle?"]
'''"What do hummingbirds eat?",
  "When do hummingbirds nest?",
  "Does a hummingbird find flowers by smell?",
  "How fast do hummingbird wings beat? How do they move?",
  "How fast do hummingbird hearts beat?",
  "How long does a hummingbird live?",
  "How many species are there?",
  "What makes a hummingbird's throat bright and colorful?",
  "Can hummingbirds walk?",
  "What is the smallest hummingbird?",
  "How many feathers do hummingbirds have?",
  "What part of a hummingbird weighs the most?",
  "How big are a hummingbird's eggs?",
  "How many breaths does a hummingbird take per minute?",
  "How far can a hummingbird fly during migration?",
  "Do hummingbirds sit on the backs of other birds during migration?",
  "Can hummingbirds smell?",
  "How do humminbirds eat nectar?",
  "How fast can a hummingbird lick?",
  "How quickly do humminbirds digest nectar?",
  "Is there cross-breeding between species?",
  "Are hummingbirds aggressive?",
  "What is the longest bill a hummingbird can have?",
  "Are hummingbirds found in the Eastern Hemisphere?",
  "What threats are posed to hummingbirds today?"]'''

best_answers = [ix + 1 for ix in range(len(test_questions))]

'''
test_questions = [
  "What is the length of time that hummingbirds will be alive?",
  "How many genera or sorts of hummingbirds can one find?",
  "Is the hue of the larynx influenced by external factors?",
  "What do they use their appendiges for and how do they move around?",
  "How little can hummingbirds get and what is the length?",
  "How much plumage do these birds have and is that amount high?",
  "What which muscle has the greatest percentage of weight?",
  "Is it true that a hummingbird lays the tiniest of ova?",
  "What is the rate of breath intake for hummingbirds?",
  "What is the farthest a hummingbird flies when it migrates?"
]

best_answers = [16 + ix for ix in range(10)]
'''
  
#final_weights =             [0.952926181485045,      0.9840099977685615, 1.0525210561258025,  1.051562827464642,   0.9532682412234448, 0.9520219911127934, 
#                             0.969117385075304,      0.9546066017400465, 1.013167700035129,   0.961371876331083,   0.9305470016082897, 0.9575960964407408, 
#                             1.0226004255054897,     0.9374376883134267, 1.0016046379331374,  1.0733357426136956,  0.9578154508191105, 0.9684130290554245, 
#                             0.9229061653172881]
#not overfit final_weights = [0.6911500010268629,     0.875678658447373,  1.162122041198649,   1.2292630528738813,  0.5251180138675038,   0.5810477163963434, 
#                             0.012372877570106517,   0.8156304084789193, 0.8917677665853175,  0.10444716164587915, 0.5033900545952501,   0.03509674711148438, 
#                             1.3054917153807546,     0.0490396070708341, 0.09729218449738365, 1.778839262953036,   0.000743946017588541, 0.056383120769699334, 
#                             -0.0007811945603587471]
final_weights = [0.788869599639297, 1.078014637709285, 1.0005490174085232, 1.120291929242625, 0.6569221944139803, 0.6517768787727969, 0.698694107948965, 0.9588163444734824, 1.339542547764566, 0.6882589326776006, 0.7001205390293421, 0.6475316992807993, 1.4377375936505312, 0.6570927217382662, 0.6527311670217246, 1.3671962223477911, 0.5684867825707811, 0.6962084180478862, 0.702095071282418, 1.2969304904469936] #jcn

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
  all_scores = state.get_scores(weights)
  
  total = 0 
  for ix, q_score_set in enumerate(all_scores):
    total += q_score_set[state.best_choices[ix]]
  
  return 1 - (total / len(all_scores))
  
  #slot = 0
  #x = sorted([(val, key) for key, val in scores.items()], reverse=True)
  
  #for xx in x:
  #  if xx[1] == state.best_answer:
  #    break
  #  slot += 1
  
  #super simple energy function. invert the score
  #this is probably too naive, but we will try it.
  #return (1 - scores[state.best_answer])# * slot / len(x)

class State(object):
  def __init__(self, qs_features, as_features, weights, best_choices):
    self.weights = weights #never used
    self.best_choices = best_choices
    self.score_vectors = []
    
    self.faq_feat = as_features
    for qf in qs_features:
      list_of_score_vectors = []
      for af in as_features:
      
        jcn_feature = []
        for qfsyn in qf.synsets:
          if qfsyn.pos() == 'v':
            for afsyn in af.synsets:
              if afsyn.pos() == 'v':
                similarity = qfsyn.jcn_similarity(afsyn, brown_ic)
                if similarity > 1: #identical words are 1e+300 and it's causing infiniti errors
                  similarity = 1 #1 is the greatest a similarity can be because of the normalization
                jcn_feature.append(similarity)
        jcn_normalize = [1] * len(jcn_feature)
        
        jcn_result = 0
        if len(jcn_feature) > 0:
          jcn_result = np.linalg.norm(jcn_feature) / np.linalg.norm(jcn_normalize)
      
        qa_score_vector = [get_score_simple(qf.tokens, af.tokens),
                           get_score_simple(qf.tokens_no_stops, af.tokens_no_stops),
                           get_score_simple(qf.lemmas, af.lemmas),
                           get_score_simple(qf.stems, af.stems),
                           get_score_simple(qf.pos_tags, af.pos_tags),
                           get_score_simple(qf.synset_lemmas, af.synset_lemmas),
                           get_score_simple(qf.antonym_lemmas, af.antonym_lemmas),
                           get_score_simple(qf.hyponym_lemmas, af.hyponym_lemmas),
                           get_score_simple(qf.hypernym_lemmas, af.hypernym_lemmas),
                           get_score_simple(qf.part_meronym_lemmas, af.part_meronym_lemmas),
                           get_score_simple(qf.part_holonym_lemmas, af.part_holonym_lemmas),
                           get_score_simple(qf.member_meronym_lemmas, af.member_meronym_lemmas),
                           get_score_simple(qf.member_holonym_lemmas, af.member_holonym_lemmas),
                           get_score_simple(qf.substance_meronym_lemmas, af.substance_meronym_lemmas),
                           get_score_simple(qf.substance_holonym_lemmas, af.substance_holonym_lemmas),
                           get_score_simple(qf.wn_definitions, af.wn_definitions),
                           get_score_simple(qf.wn_examples, af.wn_examples),
                           get_score_simple(qf.depgraph_deps, af.depgraph_deps),
                           get_score_simple(qf.depgraph_rels, af.depgraph_rels),
                           jcn_result]
        list_of_score_vectors.append(qa_score_vector)
      self.score_vectors.append(list_of_score_vectors)
      
  def get_scores(self, used_weights = None):
    scores = []
    for sv in self.score_vectors:
      specific_q_scores = dict()
      for jx, subv in enumerate(sv):
        effective_score = b.score_features(subv, used_weights)
        specific_q_scores[jx + 1] = effective_score
      scores.append(specific_q_scores)
    return scores
 
  def get_final_scores(self, used_weights = None):
    scores = []
    for sv in self.score_vectors:
      specific_q_scores = dict()
      for jx, subv in enumerate(sv):
        effective_score = b.score_features(subv, used_weights)
        specific_q_scores[self.faq_feat[jx].qapair] = effective_score
      scores.append(specific_q_scores)
    return scores
 
def lt_default(a, b): return a < b

def get_score_simple(arr1, arr2):
  if len(arr1) == 0 or len(arr2) == 0:
    return 0
  math_vecs = b.get_math_vectors(arr1, arr2, lt_default)
  return b.cosine_similarity(math_vecs[0], math_vecs[1])

question = "What do hummingbirds eat?"#"What is the lifecycle of a hummingbird like as it grows from birth as a child to death?"#"Describe the hummingbird's lifecycle."#"What do hummingbirds eat?"#"At what speed do hummingbirds fly in the air?"

def get_question_features(qfeat):
  for qf in qfeat:
    qf.synsets = lesk.get_synsets_from_features(qf)
  for qf in qfeat:
    qf.depgraphs = [dg for dg in dependency_parser.raw_parse(question)] #TODO: not the best way to do this. also iter to make an array

  for qf in qfeat:
    qf.add_wordnet_features()
    qf.add_depgraph_features()


def get_faq_features(faqs):
  as_features = b.get_answers_features(faqs)

  #this should set up the synsets
  b.load_all_synsets(as_features)

  #this should set up dependency graphs
  b.load_all_depgraphs(as_features)

  for f in as_features:
    f.add_wordnet_features()
    f.add_depgraph_features()

  return as_features

def train_model(faqs):
  feature_count = 20 #TODO: hardcoded
  learned_weights = [1] * feature_count
  qs_features = [b.TextFeatureExtraction(q, base_objects.QAPair(q,"")) for q in test_questions]
  get_question_features(qs_features)
  faq_features = get_faq_features(faqs)
  '''
  We train only once and save the weights
  Uncomment this if you want to change algo and train again.
  '''

  max_steps = 10000#25000
  state = State(qs_features, faq_features, learned_weights, best_answers)#10)#11) #5)
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
  print(state.weights)
  
  return state
