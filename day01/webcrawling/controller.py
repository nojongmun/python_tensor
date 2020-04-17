from webcrawling.entity import Entity
from webcrawling.service import Service

class Controller:
    def __init__(self):
        self.entity = Entity()
        self.service = Service()


    def bugs_music(self, url):
        self.entity.url = url
        self.entity.parser = 'lxml'
        self.service.bugs_music(self.entity)


    def naver_movie(self, url):
        self.entity.url = url
        self.entity.parser = 'html.parser'
        self.entity.path = "./data/chromedriver.exe"
        self.service.naver_movie(self.entity)