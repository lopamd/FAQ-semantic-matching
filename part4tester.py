import better_objects as b
import faq_config
import random

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
  
  #super simple energy function. invert the score
  #this is probably too naive, but we will try it.
  return 1 - scores[state.best_answer]
  
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
                      get_score_simple(q_features.pos_tags, af.pos_tags)]
      self.score_vectors.append(score_vector)
      
  def get_scores(self, used_weights = None):
    scores = dict()
    for ix, sv in enumerate(self.score_vectors):
      effective_score = b.score_features(sv, used_weights)
      scores[ix + 1] = effective_score
    return scores
                      
def lt_default(a, b): return a < b

def get_score_simple(arr1, arr2):
  math_vecs = b.get_math_vectors(arr1, arr2, lt_default)
  return b.cosine_similarity(math_vecs[0], math_vecs[1])

faqs = faq_config.getFAQs()
question = "At what speed do hummingbirds fly in the air?"

as_features = b.get_answers_features(faqs)
q_features = b.TextFeatureExtraction(question)

learned_weights = [1, 1, 1, 1, 1]

max_steps = 30000
state = State(q_features, as_features, learned_weights, 5)
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

#def get_scores(weights):
#  scores = []

#  for ix, af in enumerate(as_features):
#    score_vector = [get_score_simple(q_features.tokens, af.tokens),
#                    get_score_simple(q_features.tokens_no_stops, af.tokens_no_stops),
#                    get_score_simple(q_features.lemmas, af.lemmas),
#                    get_score_simple(q_features.stems, af.stems),
#                    get_score_simple(q_features.pos_tags, af.pos_tags)]

#    score = b.score_features(score_vector, weights)
    
#    scores.append((score, ix + 1))
    
#  return sorted(scores, reverse=True)

#for pair in sorted(scores, reverse=True):
#  print("%2d: %f" % (pair[1], pair[0]))\