import os

CONFIG_ALGO_BOW = 1
CONFIG_ALGO_NLP = 2
default_locale = 'english'

faq_input_file = 'input/Hummingbirds.csv'
evaluation_input_file = 'input/evaluationInput.csv'

path_to_stanford_lib = r'deps/stanford-corenlp-full-2018-02-27'
path_to_stanford_jar = path_to_stanford_lib + r'/stanford-corenlp-3.9.1.jar'
path_to_stanford_models_jar = path_to_stanford_lib + r'/stanford-corenlp-3.9.1-models.jar'