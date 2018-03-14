import base_objects
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

default_locale = 'english'

class NLTKFeatureExtraction( base_objects.QAFeatureExtraction ):
    def __init__( self, qa_pairs ):
        super().__init__(qa_pairs)
        
    def _tokenize( self ):
        self._tokens = []
        stops = set(nltk.corpus.stopwords.words(default_locale))
        for qa in self.qa_pairs:
            question_tokens = [w for w in nltk.word_tokenize(qa.question) if w not in stops]
            answer_tokens = [w for w in nltk.word_tokenize(qa.answer) if w not in stops]
            self._tokens.append((question_tokens, answer_tokens))
            
    def _lemmatize( self ):
        self._lemmas = []
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        for tokenpair in self.tokens:
            question_lemmas = [lemmatizer.lemmatize(x) for x in tokenpair[0]]
            answer_lemmas = [lemmatizer.lemmatize(x) for x in tokenpair[1]]
            self._lemmas.append((question_lemmas, answer_lemmas))
            
    def _stem( self ):
        self._stems = []
        stemmer = nltk.stem.PorterStemmer()
        for tokenpair in self.tokens:
            question_stems = [stemmer.stem(x) for x in tokenpair[0]]
            answer_stems = [stemmer.stem(x) for x in tokenpair[1]]
            self._stems.append((question_stems, answer_stems))
            
    def _pos_tag( self ):
        self._pos_tags = []
        for tokenpair in self.tokens:
            question_pos_tags = nltk.pos_tag(tokenpair[0])
            answer_pos_tags = nltk.pos_tag(tokenpair[1])
            self._pos_tags.append((question_pos_tags, answer_pos_tags))