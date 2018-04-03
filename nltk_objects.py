import base_objects
import nlp_config
import nltk
from collections import Counter
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
        
def get_synset_name(lemma, pos, num=1):
    return "%s.%s.%02d" % (lemma, pos, num)

class NLTKFeatureExtraction( base_objects.QAFeatureExtraction ):
    def __init__( self, qa_pairs ):
        super().__init__(qa_pairs)
    
    '''
    TODO: We may need to refactor rest of the functions as well if we have extract the features for given quesiton
          The other approach is to not make answer mandatory so that we can extract features only for questions!
    '''
    def _tokenize( self ):
        self._tokens = []
        stops = set(nltk.corpus.stopwords.words(nlp_config.default_locale))
        for qa in self.qa_pairs:
            question_tokens = [w for w in nltk.word_tokenize(qa.question) if w not in stops]
            answer_tokens = [w for w in nltk.word_tokenize(qa.answer) if w not in stops]
            self._tokens.append((question_tokens, answer_tokens))

    def _get_bow( self ):
         self._bow = []
         for tokenpair in self.tokens:
            #FIXME: We need to create one bow for both q & a ??
            self._bow.append(Counter(tokenpair[0] + tokenpair[1]))

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
    
    #the result of this function is 2-tuples of arrays of arrays of synsets. each inner array is one sentence.
    #the first outer array in a 2-tuple is for the questions. the second outer array is for the answers.
    #
    #to get the rest of the relations, you can use:
    #  syn.hypernyms()
    #     .hyponyms()
    #     .part_meronyms()
    #     .substance_meronyms()
    #     .part_holonyms()
    #     .substance_holonyms()
    def _get_synsets( self ):
        self._synsets = []
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        for qa_pos_tags in self.pos_tags:
            #qa_pos_tags[0] is an array of arrays of pos tags for the question sentences. ('constructor', 'NN')
            #qa_pos_tags[1] is an array of arrays of pos tags for the answers sentences.
            
            def get_synsets_for_pos_tags(a_pos_tags):
                ret_synsets = []
                for word_pos_pair in a_pos_tags:
                    wordnet_pos = penn2morphy(word_pos_pair[1])
                    if wordnet_pos:
                        try:
                            ret_synsets.append(wn.synset(get_synset_name(lemmatizer.lemmatize(word_pos_pair[0]), wordnet_pos)))
                        except:
                            ret_synsets.append(None) #not sure if we should append none or just pass
                return ret_synsets
            
            q_sentence_synsets = [get_synsets_for_pos_tags(sentence_tags) for sentence_tags in qa_pos_tags[0]]    
            a_sentence_synsets = [get_synsets_for_pos_tags(sentence_tags) for sentence_tags in qa_pos_tags[1]]
            
            self._synsets.append((q_sentence_synsets, a_sentence_synsets))