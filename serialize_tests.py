import nltk_objects as no
import faq_config

sub_folder = 'data/depgraphs'

faqs = faq_config.getFAQs()
feature_extractor = no.NLTKFeatureExtraction(faqs)

def save_dependency_graphs(graphs, filename):
    with open(filename, "w+") as outfile:
        first = True
        for graph in graphs:
            if not first:
                outfile.write('\n')
            outfile.write(graph.to_conll(4))
            first = False
            
def extract_graphs(alist):
    ret = []
    for iter in alist:
        ret.extend([x for x in iter])
    return ret

faq_number = 1
         
for faq_graphs in feature_extractor.dependency_graphs:
    q_graphs = faq_graphs[0] #these are lists of list iterators
    a_graphs = faq_graphs[1] # because parsing returns a list iterator
    
    save_dependency_graphs(extract_graphs(q_graphs), "%s/%d_question.conll" % (sub_folder, faq_number))
    save_dependency_graphs(extract_graphs(a_graphs), "%s/%d_answer.conll" % (sub_folder, faq_number))
    
    faq_number += 1