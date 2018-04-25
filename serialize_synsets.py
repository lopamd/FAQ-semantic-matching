import nltk_objects as no
import faq_config
import lesk

#shoutout to Avicii

#TODO: not sure if i need to remove stopwords before lemmatizing (ok, tokenizing does that)
#      but i'm not sure if tokenizing should do that........

#TODO: some words (like In) may need to be lowercased

#TODO: maybe we should leave stopwords. like "to" should be there for verbs i feel...

#TODO: words like "United States" are being tagged with synsets separately

sub_folder = 'data/synsets'

faqs = faq_config.getFAQs()
feature_extractor = no.NLTKFeatureExtraction(faqs)

#flatten = lambda l: [item for sublist in l for item in sublist]

def save_synsets(lemmas, filename):
    with open(filename, "w+") as outfile:
        first = True
        for lemma in lemmas:
            lemma_synset = lesk.get_lemma_synset(lemma, lemmas)
            
            if lemma_synset is not None:
                if not first:
                    outfile.write('\n')
                outfile.write("%s %s" % (lemma, lemma_synset.name()))
                first = False

faq_number = 1
         
for faq_lemmas in feature_extractor.lemmas:
    q_lemmas = faq_lemmas[0]
    a_lemmas = faq_lemmas[1]
    
    save_synsets(q_lemmas, "%s/%d_question.txt" % (sub_folder, faq_number))
    save_synsets(a_lemmas, "%s/%d_answer.txt" % (sub_folder, faq_number))
    
    faq_number += 1