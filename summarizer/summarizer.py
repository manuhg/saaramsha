import os
from time import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import spacy


class summarizer:
    def __init__(self, encoder_name=None):
        self.encoder_name = encoder_name
        self.embedding_funcs = {'spacy':self.get_sent_embeds_spacy,'USE':self.get_sent_embeds_USE}
        self.nlp = None
        self.universal_sent_encoder = None
        self.get_sent_embeds = self.embedding_funcs.get(encoder_name,self.get_sent_embeds_USE)
    
    def setup_env(self):
        print('Setting up env... ')
        print(os.popen('pip install -q tensorflow_hub').read())
        print(os.popen('pip install -q spacy').read())
        print(os.popen('python -m spacy download en_core_web_md').read())

    def load_encoders(self):
        print('Loading sentence encoders...')
        if self.encoder_name=='spacy':
            self.nlp = spacy.load('en_core_web_lg')
        elif self.encoder_name=='USE':
            self.universal_sent_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        else:
            self.nlp = spacy.load('en_core_web_lg')
            self.universal_sent_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

    def load(self):
        print('Loading...')
        try:
            self.load_encoders()
        except Exception as e:
            self.setup_env()
            self.load_encoders()
        print('Done Loading')
    
    def get_sent_embeds_spacy(self,doc):
        return np.array(list(map(lambda sent:self.nlp(sent).vector,doc)))
    
    def get_sent_embeds_USE(self,doc):#universal sentence encoder
        init = tf.global_variables_initializer()
        table_init = tf.tables_initializer()
        
        sess = tf.Session()
        with sess.as_default():
            sess.run([init, table_init])
            return sess.run(self.universal_sent_encoder(doc))
    
    def summarize(self,document,encoder_name='USE'):
        self.get_sent_embeds = self.embedding_funcs.get(encoder_name,self.get_sent_embeds_USE)
        t1 = time()
        print('Splitting into sentences...',round(time()-t1,4))
        doc = sent_tokenize(document)
        print('No of sentences :',len(doc))
        print('Starting to encode...',round(time()-t1,4))
        enc_doc = self.get_sent_embeds(doc)
        print('Encoding Finished',round(time()-t1,4))
        n_clusters = int(np.ceil(len(enc_doc)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans = kmeans.fit(enc_doc)
        avg = []
        closest = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, enc_doc)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([doc[closest[idx]] for idx in ordering])
        print('Done',round(time()-t1,4))
        return summary

if __name__ == "__main__":
    document = 'The encoder-decoder architecture is used as a way of building RNNs for sequence predictions. It involves two major components: an encoder and a decoder. The encoder reads the complete input sequence and encodes it into an internal representation, usually a fixed-length vector, described as the context vector. The decoder, on the other hand, reads the encoded input sequence from the encoder and generates the output sequence. Various types of encoders can be used more commonly, bidirectional RNNs, such as LSTMs, are used.'
    s = summarizer()
    s.load()
    summary = s.summarize(document)
    print('\n\nOriginal:\n', document, '\nSummary:\n', summary)
