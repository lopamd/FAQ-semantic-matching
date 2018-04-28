import nltk_objects
import faq_config
import base_objects
import operator

from nlp_algo import BOWAlgorithm
from collections import defaultdict
import better_objects as b
import part4tester as model

TOPTEN_ANS = []

class Evaluation(object):

    def __init__(self, algoType,fext):
        self.scores = 0
        self.count = 0
        self.aType = algoType
        self.rdict = defaultdict()
        self.fext = fext

    def get_topNResults(self, resultDict, n):
        sortedResults = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
        count = 0
        for qa_pair,score in sortedResults:
            if count < n:
                TOPTEN_ANS.append(qa_pair.answer)
                count = count + 1

    def computeResult(self):
        #For evaluation of BOW
        eval_qns = faq_config.getEvaluationQns()
        for qns in eval_qns:
            TOPTEN_ANS.clear()
            user_qa = [base_objects.QAPair(qns.question, "")]

            if self.aType == 1:
                #BOW Type
                user_feat_extractor = nltk_objects.NLTKFeatureExtraction(user_qa)
                bow_algo = BOWAlgorithm(user_qa, user_feat_extractor, self.fext)
                resultDict = bow_algo._compute()
            else:
                uq_nlp_feat = [b.TextFeatureExtraction(qns.question, qns)]
                tstate = model.State(uq_nlp_feat, self.fext, model.final_weights, None)
                resultDict = tstate.get_final_scores(model.final_weights)[0]
            self.get_topNResults(resultDict, 10)
            index_ = TOPTEN_ANS.index(qns.answer) if qns.answer in TOPTEN_ANS else -1
            print ("Question is: ",qns.question)
            print ("Correct answer at index: ", index_)
            print ("--------------------------------------------")
            self.rdict.update({qns.question : index_+1})

class MRREvaluation(Evaluation):

    def __init__(self, algoType, fext):
        super().__init__(algoType, fext)

    def get_rr(self):
        i = 0
        for key, value in self.rdict.items():
            if value != 0:
                rr = 1.0 / float(value)
                print (key, rr)
                self.scores += rr
            else:
                print (key, 0)
            self.count += 1
    # mrr
    def get_mrr(self):
        return self.scores / self.count