import base_objects
import faq_config

faqs = faq_config.getFAQs()
feature_extractor = base_objects.NLTKFeatureExtraction(faqs)

for qatoken in feature_extractor.tokens:
    print(qatoken)

for qalemma in feature_extractor.lemmas:
    print(qalemma)
    
for qastem in feature_extractor.stems:
    print(qastem)