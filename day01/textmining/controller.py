from textmining.entity import Entity
from textmining.service import Service
import  nltk
class Controller:

    def __init__(self):
        self.entity = Entity()
        self.service = Service()

    def download_dictionary(self):
        nltk.download('all')


    def data_analysis(self):
        self.entity.fname = 'kr-Report_2018.txt'
        self.entity.context = './data/'
        self.service.extract_tokens(self.entity)
        self.service.extract_hangeul()
        self.service.conversion_token()
        self.service.compound_noun()
        self.entity.fname = 'stopwords.txt'
        self.service.extract_stopword(self.entity)
        self.service.filtering_text_with_stopword()
        self.service.frequent_text()
        self.entity.fname ="D2Coding.ttf"
        self.service.draw_worldcloud(self.entity)


if __name__ == '__main__':
    app = Controller()
    # app.download_dictionary()
    app.data_analysis()

