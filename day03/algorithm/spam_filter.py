from sklearn.feature_extraction import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import glob
import os
import os
import numpy as np
import  nltk

class SpamFilter:
    context : str
    fname : str
    target : str
    def __init__(self):
        self.lemmatizer = None
        self.all_names = None

    @property
    def context(self) -> str: return self._context
    @context.setter
    def context(self, context): self._context = context
    @property
    def fname(self) -> str: return self._fname
    @fname.setter
    def fname(self, fname): self._fname = fname
    @property
    def target(self) -> str: return self._target
    @target.setter
    def target(self, target): self._target = target

    def email_test(self, this):
        ham = ''
        spam = ''
        with open(ham, 'r') as infile:
            ham_sample = infile.read()
        with open(spam, 'r') as infile:
            spam_sample = infile.read()
        cv = CountVectorizer(stop_words="english", max_features=500)
        emails, labels = [], []
        file_path = './ham/'
        for filename in glob.glob(os.path.join(file_path, '*.txt')):
            with open(filename, 'r', encoding='ISO-8859-1') as infile:
                emails.append(infile.read())
                labels.append(0)

        file_path = './spam/'
        for filename in glob.glob(os.path.join(file_path, '*.txt')):
            with open(filename, 'r', encoding='ISO-8859-1') as infile:
                emails.append(infile.read())
                labels.append(1)

        self.all_names = set(names.words())
        self.lemmatizer = WordNetLemmatizer()
        cleaned_emails = self.clean_text(emails)
        term_docs = cv.fit_transform(cleaned_emails)
        feature_names = cv.get_feature_names()
        print(feature_names[:5])


    def clean_text(self, docs):
        cleaned_docs = []
        for doc in docs:
            cleaned_docs.append(' '.join([self.lemmatizer.lemmatize(word.lower())
                                          for word in doc.split()
                                          if self.letters_only(word)
                                          and word not in self.all_names]))
        return cleaned_docs



if __name__ == '__main__':
    # nltk.download('all') : 한번만 받으면 된다.
    pass

























