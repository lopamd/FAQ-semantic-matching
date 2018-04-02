class QAPair:
     def __init__(self, question, answer):
         self.question = question
         self.answer = answer
         
class QAFeatureExtraction( object ):
    '''qa_pairs is a list of QAPair objects'''
    def __init__( self, qa_pairs ):
        self.qa_pairs = qa_pairs
        self._tokens = None
        self._sentence_tokens = None
        self._lemmas = None
        self._stems = None
        self._pos_tags = None
        self._dependency_graphs = None
        self._synsets = None
        self._bow = None #This will hold a List of Counter object of each FAQ

    '''Private abstract function to tokenize the questions and answers on word boundaries'''
    def _tokenize( self ):
        raise NotImplementedError("Class %s doesn't implement _tokenize()" % (self.__class__.__name__))

    '''Private abstract function to tokenize the questions and answers on sentence boundaries'''
    def _tokenize_sentences( self ):
        raise NotImplementedError("Class %s doesn't implement _tokenize_sentences()" % (self.__class__.__name__))

    '''Private abstract function to lemmatize the questions and answers'''
    def _lemmatize( self ):
        raise NotImplementedError("Class %s doesn't implement _lemmatize()" % (self.__class__.__name__))

    '''Private abstract function to stem the questions and answers'''
    def _stem( self ):
        raise NotImplementedError("Class %s doesn't implement _stem()" % (self.__class__.__name__))
        
    '''Private abstract function to pos tag the questions and answers''' 
    def _pos_tag( self ):
        raise NotImplementedError("Class %s doesn't implement _pos_tag()" % (self.__class__.__name__))
        
    '''Private abstract function to graph the dependencies for the questions and answers''' 
    def _graph_dependencies( self ):
        raise NotImplementedError("Class %s doesn't implement _graph_dependencies()" % (self.__class__.__name__))
        
    '''Private abstract function to get wordnet synsets for the lemmas in the questions and answers''' 
    def _get_synsets( self ):
        raise NotImplementedError("Class %s doesn't implement _get_synsets()" % (self.__class__.__name__))
        
    '''Private abstract function to get bag of words the questions and answers''' 
    def _get_bow( self ):
        raise NotImplementedError("Class %s doesn't implement _get_synsets()" % (self.__class__.__name__))
    
    @property
    def tokens(self):
        if self._tokens is None:
            self._tokenize()
        return self._tokens
        
    @property
    def sentence_tokens(self):
        if self._sentence_tokens is None:
            self._tokenize_sentences()
        return self._sentence_tokens
    
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
        
    @property
    def dependency_graphs(self):
        if self._dependency_graphs is None:
            self._graph_dependencies()
        return self._dependency_graphs
        
    @property
    def synsets(self):
        if self._synsets is None:
            self._get_synsets()
        return self._synsets
    @property
    def bow(self):
        if self._bow is None:
            self._get_bow()
        return self._bow