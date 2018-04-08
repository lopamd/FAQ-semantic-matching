import nltk_objects
import faq_config

faqs = faq_config.getFAQs()
feature_extractor = nltk_objects.NLTKFeatureExtraction(faqs)

for qatoken in feature_extractor.tokens:
    print(qatoken)

for qatoken in feature_extractor.sentence_tokens:
    print(qatoken)

for qabow in feature_extractor.bow:
    print(qabow)

for qalemma in feature_extractor.lemmas:
    print(qalemma)
    
for qastem in feature_extractor.stems:
    print(qastem)
   
for postag in feature_extractor.pos_tags:
    print(postag)
    
for graphs in feature_extractor.dependency_graphs:
    print(graphs)

for syns in feature_extractor.synsets:
    print(syns)

'''
Test cases:

Mandatory for Q2:
1. Exact same faq question in the input: It should return the same answer
2. Couple of words missing: It should return the same answer
3. Words jumbled or transposed: It should return the same answer
4. Synonyms or similar semantic meaing: Doesn't expect to return correct answer

Q3: Should show imporovements over Q2
TODO: Write the updated test cases here
'''