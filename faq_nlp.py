import nltk_objects
import faq_config
import base_objects







def main():

    print("****** Hummingbird FAQ engine powered by NLTK *********")
    faqs = faq_config.getFAQs()
    feature_extractor = nltk_objects.NLTKFeatureExtraction(faqs)
    qatokens = feature_extractor.tokens
    qabows = feature_extractor.bow

    '''
    for bow in qabows:
        print(bow) 
    '''

    user_q = input("Input your question:")
    
    #FIXME: It has to be added to the empty list because nltk_objct operats on the list
    #Alt: Alternate approach. Only call __tokenize(). But move stops to a class variable.

    user_qa = [base_objects.QAPair(user_q, "")]
    user_feat_extractor = nltk_objects.NLTKFeatureExtraction(user_qa)
    print(user_feat_extractor.tokens)
    print(user_feat_extractor.bow)
 
 

if __name__ == "__main__":
    main()
