from models import summarizer_sk


class summarizer:
    def __init__(self, model=''):
        pass

    def load(self):
        summarizer_sk.load_model()

    def summarize(self, document_str):
        documents = [document_str]  # can send multiple documents at once
        # returned variable is an array since if multiple docs are sent
        return summarizer_sk(documents)[0]
