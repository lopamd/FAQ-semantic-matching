class NLPAlgorithm:
    def __init__(self, uquestion):
        self.user_question = uquestion
        
        #This will contain the evaluation score
        self.score = None

        #Final results
        self.results = None
        
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
    def __init__(self, uquestion):
        super().__init__(uquestion)

    def _compute(self):
        pass
    
    def _evaluate(self):
        pass
