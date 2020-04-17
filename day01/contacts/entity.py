class Entity:
    def __init__(self):
        self._name = ''
        self._phone = ''
        self._email = ''
        self._addr = ''

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def phone(self) -> str:
        return self._phone

    @phone.setter
    def phone(self, phone):
        self._phone = phone

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, email):
        self._email = email

    @property
    def addr(self) -> str:
        return self._addr

    @addr.setter
    def addr(self, addr):
        self._addr = addr

    def to_string(self):
        return f'이름{self._name}, 전화번호{self._phone}, 이메일{self._email}, 주소{self._addr}'
