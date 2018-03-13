class QAPair:
     def __init__(self, question, answer):
         self.question = question
         self.answer = answer
         
class QAFeatureExtraction( object ):
    '''qa_pairs is a list of QAPair objects'''
    def __init__( self, qa_pairs ):
        self.qa_pairs = qa_pairs

    '''Private abstract function to tokenize the questions and answers'''
    def __tokenize( self ):
        raise NotImplementedError("Class %s doesn't implement tokenize()" % (self.__class__.__name__))

    '''Private abstract function to lemmatize the questions and answers'''
    def __lemmatize( self ):
        raise NotImplementedError("Class %s doesn't implement lemmatize()" % (self.__class__.__name__))

    '''Private abstract function to stem the questions and answers'''
    def __stem( self ):
        raise NotImplementedError("Class %s doesn't implement stem()" % (self.__class__.__name__))