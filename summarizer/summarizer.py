from models import summarizer_sk
class summarizer:
    def __init__(self, model=''):
        pass

    def load(self):
        summarizer_sk.load_model()

    def summarize(self, document_str):
        documents = [document_str]  # can send multiple documents at once
        # returned variable is an array since if multiple docs are sent
        return summarizer_sk.summarize(documents)[0]


if __name__ == "__main__":
    document = 'The encoder-decoder architecture is used as a way of building RNNs for sequence predictions. It involves two major components: an encoder and a decoder. The encoder reads the complete input sequence and encodes it into an internal representation, usually a fixed-length vector, described as the context vector. The decoder, on the other hand, reads the encoded input sequence from the encoder and generates the output sequence. Various types of encoders can be used more commonly, bidirectional RNNs, such as LSTMs, are used.'
    s = summarizer()
    s.load()
    summary = s.summarize(document)
    print('\n\nOriginal:\n', document, '\nSummary:\n', summary)
