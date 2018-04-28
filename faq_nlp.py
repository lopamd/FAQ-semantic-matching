import nltk_objects
import faq_config
import base_objects
import operator

from nlp_algo import BOWAlgorithm
from nlp_eval import MRREvaluation
import better_objects as b
import part4tester as model

RESULTS_TOPN = 10


def print_results(user_q, resultDict, algoType):
    sortedResults = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
    count = 0
    print("***********************************************************************")
    print("Given user question: ", user_q)
    print("***********************************************************************")
    if (algoType == 1):
        print("Top 10 results from Bag of words algorithm are:")
    else:
        print("Top 10 results NLP Pipeline algorithm are:")
    for qa_pair,score in sortedResults:
        if count < RESULTS_TOPN:
            print(qa_pair.answer,score)
            count = count + 1
def print_eval_result(evalObj, algoType):

    if algoType == 1:
        algoName = "BagOfWords"
    else:
        algoName = "NLP Pipeline"
    print("***********************************************************************")
    print("MRR EVALUATION for algorithm: ", algoName)
    print("***********************************************************************")
    print (evalObj.get_rr())
    print ('------------------------------------------------------------')
    print ("Total MRR of the QA Set: ",evalObj.get_mrr())

def main():

    print("****** Hummingbird FAQ engine powered by NLTK *********")
 
    faqs = faq_config.getFAQs()
    faq_bow_feat = nltk_objects.NLTKFeatureExtraction(faqs)
    faq_nlp_feat = model.get_faq_features(faqs)
    '''
    algoType = 1
    evaluation = MRREvaluation(algoType, feature_extractor)
    evaluation.computeResult()
    print_eval_result(evaluation, algoType)
    '''
    
    #user_q = input("Input your question:")
    #user_q = "when is hummingbird season"
    user_q = "Does hummingbirds migrate in winter?"
    #user_q = "How fast do hummingbirds' wings beat per second?"

    if user_q == "" or user_q == None:
        raise ValueError("Invalid question given. Exiting")
        exit(1)
 
    #FIXME: It has to be added to the empty list because nltk_object operates on the list
    #Alt: Alternate approach. Only call __tokenize(). But move stops to a class variable.

    user_qa = [base_objects.QAPair(user_q, "")]
    uq_bow_feat = nltk_objects.NLTKFeatureExtraction(user_qa)
    uq_nlp_feat = [b.TextFeatureExtraction(user_q, user_qa)]
    #print(user_feat_extractor.tokens)
    #print(user_feat_extractor.bow)

    #BOW specific implementation.
    algoType = 1
    bow_algo = BOWAlgorithm(user_q, uq_bow_feat, faq_bow_feat)
    resultDict = bow_algo._compute()
    print_results(user_q, resultDict,algoType)
 
    #NLP Pipeline specific
 
    '''
    TRAINING Code
    ''' 
    #model.train_model(faqs)

    '''
    Testing code
    '''
    algoType = 2
    tstate = model.State(uq_nlp_feat, faq_nlp_feat, model.final_weights, None)
    nlp_rdict = tstate.get_final_scores(model.final_weights)
    print_results(user_q, nlp_rdict[0], algoType)

if __name__ == "__main__":
    main()