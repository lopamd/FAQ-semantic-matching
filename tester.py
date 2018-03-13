import base_objects
import faq_config

faqs = faq_config.getFAQs()
feature_extractor = base_objects.NTLKFeatureExtraction(faqs)

for qatokens in feature_extractor.tokens:
    print(qatokens)