import nltk_objects
import faq_config
import base_objects
import operator

from nlp_algo import BOWAlgorithm

RESULTS_TOPN = 10

def print_results(user_q, resultDict):
    sortedResults = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
    count = 0

    print("***********************************************************************")
    print("Given user question: ", user_q)
    print("***********************************************************************")
    print("Top 10 results from Bag of words algorithm are:")
    for qa_pair,score in sortedResults:
        if count < RESULTS_TOPN:
            print(qa_pair.answer,score)
            count = count + 1

def main():

    print("****** Hummingbird FAQ engine powered by NLTK *********")
 
    faqs = faq_config.getFAQs()
    feature_extractor = nltk_objects.NLTKFeatureExtraction(faqs)

    user_q = input("Input your question:")
    #user_q = "when is hummingbird season"
    #user_q = "Where do hummingbirds go in the winter?"
    #user_q = "How fast do hummingbirds' wings beat per second?"

    if user_q == "" or user_q == None:
        raise ValueError("Invalid question given. Exiting")
        exit(1)
 
    #FIXME: It has to be added to the empty list because nltk_object operates on the list
    #Alt: Alternate approach. Only call __tokenize(). But move stops to a class variable.

    user_qa = [base_objects.QAPair(user_q, "")]
    user_feat_extractor = nltk_objects.NLTKFeatureExtraction(user_qa)
    #print(user_feat_extractor.tokens)
    #print(user_feat_extractor.bow)

    '''
    BOW specific implementation.
    '''
    bow_algo = BOWAlgorithm(user_q, user_feat_extractor, feature_extractor)
    resultDict = bow_algo._compute()
    print_results(user_q, resultDict)
    
    #TODO: Now add the NLP engine algorithm

if __name__ == "__main__":
    main()