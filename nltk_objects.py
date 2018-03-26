import base_objects
import nlp_config
import nltk
from nltk.parse.stanford import StanfordDependencyParser
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

dependency_parser = StanfordDependencyParser(path_to_jar=nlp_config.path_to_stanford_jar, path_to_models_jar=nlp_config.path_to_stanford_models_jar)

class NLTKFeatureExtraction( base_objects.QAFeatureExtraction ):
    def __init__( self, qa_pairs ):
        super().__init__(qa_pairs)
        
    def _tokenize( self ):
        self._tokens = []
        stops = set(nltk.corpus.stopwords.words(nlp_config.default_locale))
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
    
    def _graph_dependencies( self ):
        self._dependency_graphs = []
        for qa in self.qa_pairs:
            question_graph = dependency_parser.raw_parse(qa.question)
            answer_graph = dependency_parser.raw_parse(qa.answer)
            self._dependency_graphs.append((question_graph, answer_graph))