import base_objects
import faq_config

faqs = faq_config.getFAQs()
feature_extractor = base_objects.NLTKFeatureExtraction(faqs)

for qatokens in feature_extractor.tokens:
    print(qatokens)

for qalemmas in feature_extractor.lemmas:
    print(qalemmas)