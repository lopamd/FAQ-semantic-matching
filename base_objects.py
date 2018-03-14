class QAPair:
     def __init__(self, question, answer):
         self.question = question
         self.answer = answer
         
class QAFeatureExtraction( object ):
    '''qa_pairs is a list of QAPair objects'''
    def __init__( self, qa_pairs ):
        self.qa_pairs = qa_pairs
        self._tokens = None
        self._lemmas = None
        self._stems = None
        self._pos_tags = None

    '''Private abstract function to tokenize the questions and answers'''
    def _tokenize( self ):
        raise NotImplementedError("Class %s doesn't implement _tokenize()" % (self.__class__.__name__))

    '''Private abstract function to lemmatize the questions and answers'''
    def _lemmatize( self ):
        raise NotImplementedError("Class %s doesn't implement _lemmatize()" % (self.__class__.__name__))

    '''Private abstract function to stem the questions and answers'''
    def _stem( self ):
        raise NotImplementedError("Class %s doesn't implement _stem()" % (self.__class__.__name__))
        
    '''Private abstract function to pos tag the questions and answers''' 
    def _pos_tag( self ):
        raise NotImplementedError("Class %s doesn't implement _pos_tag()" % (self.__class__.__name__))
        
    @property
    def tokens(self):
        if self._tokens is None:
            self._tokenize()
        return self._tokens
    
    @property
    def lemmas(self):
        if self._lemmas is None:
            self._lemmatize()
        return self._lemmas
        
    @property
    def stems(self):
        if self._stems is None:
            self._stem()
        return self._stems
        
    @property
    def pos_tags(self):
        if self._pos_tags is None:
            self._pos_tag()
        return self._pos_tags