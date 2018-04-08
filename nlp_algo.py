from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

class NLPAlgorithm:
    def __init__(self, uquestion, qfeat, docfeat):
        self.user_question = uquestion
        #This will contain the qapair:score
        self.scoreDict = defaultdict()
        self.qafeat = qfeat
        self.docs_feature = docfeat
        #Final results
        self.results = list()
    '''Private abstract function to implement the actual algorithm'''
    def _compute(self):
        raise NotImplementedError("Class %s doesn't implement _compute()" % (self.__class__.__name__))

    '''Private abstract function to evaluate the output'''
    def _evaluate(self):
        raise NotImplementedError("Class %s doesn't implement _evaluate()" % (self.__class__.__name__))

    '''Function to print the output'''
    def _print(self):
        print("Final Score is " + self.score)

class BOWAlgorithm(NLPAlgorithm):
    def __init__(self, uquestion, qfeat, docfeat):
        super().__init__(uquestion, qfeat, docfeat)

    def __compute_cosine(self, query, doc):
        query = np.array(query).reshape(1,-1)
        doc = np.array(doc).reshape(1,-1)
        return cosine_similarity(query, doc)
    def _compute(self):

        '''
        TODO: QAFeatureExtraxction object has qa_pairs and _bow in the same order.
        This works as both are sequentially accessed. So _bow index can be used to
        access corresponding qa_pair
        '''
        query_vec = list(self.qafeat.bow[0].values())
        for index, faq_bow in enumerate(self.docs_feature.bow):
            faq_vec = []
            for word in self.qafeat.bow[0]:
                if word in faq_bow:
                    faq_vec.append(faq_bow[word])
                else:
                    faq_vec.append(0)
            if len(faq_vec) != 0:
                #cosine similarity returns in numpy array. Convert it into regular val
                simScore = self.__compute_cosine(query_vec, faq_vec).tolist()[0][0]
                self.scoreDict[self.docs_feature.qa_pairs[index]] = simScore
            else:
                print("No matching words found")
        return self.scoreDict
    def _evaluate(self):
       #Use the Evaluation objects here
 
       pass
