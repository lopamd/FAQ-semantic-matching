import nltk_objects
import faq_config
import base_objects
import operator

from nlp_algo import BOWAlgorithm
from nlp_eval import MRREvaluation
from nlp_config import *
import better_objects as b
import part4tester as model

RESULTS_TOPN = 10

do_training = False
report_training = False
do_main = True

def print_results(user_q, resultDict, algoType):
    sortedResults = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
    count = 0
    print("***********************************************************************")
    print("Given user question: ", user_q)
    print("***********************************************************************")
    if (algoType == CONFIG_ALGO_BOW):
        print("Top 10 results from Bag of words algorithm are:")
    else:
        print("Top 10 results NLP Pipeline algorithm are:")
    for qa_pair,score in sortedResults:
        if count < RESULTS_TOPN:
            print(qa_pair.answer,score)
            count = count + 1
def print_eval_result(evalObj, algoType):

    if algoType == CONFIG_ALGO_BOW:
        algoName = "BagOfWords"
    else:
        algoName = "NLP Pipeline"
    print("***********************************************************************")
    print("MRR EVALUATION for algorithm: ", algoName)
    print("***********************************************************************")
    print (evalObj.get_rr())
    print ('------------------------------------------------------------')
    print ("Total MRR of the QA Set: ",evalObj.get_mrr())

def run_mrr(faq_feat,algoType):
    evaluation = MRREvaluation(algoType, faq_feat)
    evaluation.computeResult()
    print_eval_result(evaluation, algoType)

def run_userq(user_qa, faq_feat, algoType):

    #FIXME: It has to be added to the empty list because nltk_object operates on the list
    #Alt: Alternate approach. Only call __tokenize(). But move stops to a class variable.
    user_q = user_qa[0].question
    if (algoType == CONFIG_ALGO_BOW):
        #BOW specific implementation.
        uq_bow_feat = nltk_objects.NLTKFeatureExtraction(user_qa)
        bow_algo = BOWAlgorithm(user_q, uq_bow_feat, faq_feat)
        resultDict = bow_algo._compute()
    else:
        #NLP Pipeline specific
        uq_nlp_feat = [b.TextFeatureExtraction(user_q, user_qa)]

        '''
        Testing code
        '''

        tstate = model.State(uq_nlp_feat, faq_feat, model.final_weights, None)
        nlp_rdict = tstate.get_final_scores(model.final_weights)
        resultDict = nlp_rdict[0]

    print_results(user_q, resultDict, algoType)

def space_out():
  print()
  print()
  print()
    
def main():

    print("****** Hummingbird FAQ engine powered by NLTK *********")

    faqs = faq_config.getFAQs()

    '''
    TRAINING Code
    '''
    if do_training:
      state = model.train_model(faqs)
      model.final_weights = state.weights
      
      if report_training:
        all_scores = state.get_scores(state.weights)
        for ix, q_score_set in enumerate(all_scores):
          dict_scores = sorted([(ascore, qnum) for qnum, ascore in q_score_set.items()], reverse=True)
          print(state.best_choices[ix])
          for pair in dict_scores:
            print("%2d: %f" % (pair[1], pair[0]))
          print()
    
    if do_main:      
      faq_bow_feat = nltk_objects.NLTKFeatureExtraction(faqs)
      faq_nlp_feat = model.get_faq_features(faqs)

      run_mrr(faq_bow_feat, CONFIG_ALGO_BOW)
      
      space_out()
      
      run_mrr(faq_nlp_feat, CONFIG_ALGO_NLP)
      
      #'''
      #user_q = input("Input your question:")
      #user_q = "when is hummingbird season"
      user_q = "Do hummingbirds migrate in winter?"
      #user_q = "How fast do hummingbirds' wings beat per second?"

      if user_q == "" or user_q == None:
          raise ValueError("Invalid question given. Exiting")
          exit(1)
      user_qa = [base_objects.QAPair(user_q, "")]

      space_out()
      
      run_userq(user_qa, faq_bow_feat, CONFIG_ALGO_BOW)
      
      space_out()
      
      run_userq(user_qa, faq_nlp_feat, CONFIG_ALGO_NLP)
      
if __name__ == "__main__":
    main()