import nltk_objects
import faq_config

faqs = faq_config.getFAQs()
feature_extractor = nltk_objects.NLTKFeatureExtraction(faqs)

for qatoken in feature_extractor.tokens:
    print(qatoken)

for qalemma in feature_extractor.lemmas:
    print(qalemma)
    
for qastem in feature_extractor.stems:
    print(qastem)
    
for postag in feature_extractor.pos_tags:
    print(postag)