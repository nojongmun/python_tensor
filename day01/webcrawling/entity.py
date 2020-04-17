class Entity:
    def __init__(self):
        self.url = ''
        self.parser = ''
        self.path = ''
        self.api = ''
        self.apikey = ''

    @property
    def url(self) -> str:
        return self._url

    # setter 선언
    @url.setter
    def url(self, url):
        self._url = url

    @property
    def parser(self) -> str:
        return self._parser

    # setter 선언
    @parser.setter
    def parser(self, parser):
        self._parser = parser

    @property
    def path(self) -> str:
        return self._path

    # setter 선언
    @path.setter
    def path(self, path):
        self._path = path

    @property
    def api(self) -> str:
        return self._api

    # setter 선언
    @api.setter
    def api(self, api):
        self._api = api

    @property
    def apikey(self) -> str:
        return self._apikey

    # setter 선언
    @apikey.setter
    def apikey(self, apikey):
        self._apikey = apikey

    def to_string(self):
        return f'이름{self._url}, 전화번호{self._parser}, 이메일{self._api}, 주소{self._apikey}'
