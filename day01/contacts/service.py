# service는 entity의 집합체(복합체) ( entity는 유니)

class Service:

    def __init__(self):
        # 리스트 구조조
        self._contacts = []

    def add_contact(self,payload):
        self._contacts.append(payload)

    def get_contacts(self):
        contacts = []
        for i in self._contacts:
            contacts.append(i.to_string())
        return ' '.join(contacts)

    def del_contacts(self, name):
        for i, t in enumerate(self._contacts):
            if t.name == name:
                del self._contacts[i]