import base_objects
import nlp_config
import nltk
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

dependency_parser = StanfordDependencyParser(path_to_jar=nlp_config.path_to_stanford_jar, path_to_models_jar=nlp_config.path_to_stanford_models_jar)

def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

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
       
    def _tokenize_sentences( self ):
        self._sentence_tokens = []
        for qa in self.qa_pairs:
            question_sentences = nltk.sent_tokenize(qa.question)
            answer_sentences = nltk.sent_tokenize(qa.answer)
            self._sentence_tokens.append((question_sentences, answer_sentences))
            
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
        for tokenpair in self.sentence_tokens:
            question_word_tokens = [nltk.word_tokenize(sentence) for sentence in tokenpair[0]]
            answer_word_tokens = [nltk.word_tokenize(sentence) for sentence in tokenpair[1]]
            question_pos_tags = [nltk.pos_tag(sentence) for sentence in question_word_tokens]
            answer_pos_tags = [nltk.pos_tag(sentence) for sentence in answer_word_tokens]
            self._pos_tags.append((question_pos_tags, answer_pos_tags))
    
    def _graph_dependencies( self ):
        self._dependency_graphs = []
        for tokenpair in self.sentence_tokens:
            question_graph = [dependency_parser.parse(sentence) for sentence in tokenpair[0]]
            answer_graph = [dependency_parser.parse(sentence) for sentence in tokenpair[1]]
            self._dependency_graphs.append((question_graph, answer_graph))