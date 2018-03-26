import csv
import base_objects
import nlp_config

class FAQReader( object ):
    """Abstract class. Please implement fetch to return a list of QAPairs."""
    def fetch( self ):
        raise NotImplementedError("Class %s doesn't implement fetch()" % (self.__class__.__name__))
        
class CSVFAQReader( FAQReader ):
    def __init__(self, csvfilename):
        self.csvfilename = csvfilename

    def fetch( self ):
        faqs = []
        with open(self.csvfilename) as csvfile:
            areader = csv.reader(csvfile)
            for row in areader:
                faqs.append(base_objects.QAPair(row[0].strip(), row[1].strip()))
        return faqs
        
        
def getFAQs():
    faqreader = CSVFAQReader(nlp_config.faq_input_file)
    return faqreader.fetch()
  
#Usage:
#import faq_config
#for y in faq_config.getFAQs():
#    print(y.question)
#    print(y.answer)