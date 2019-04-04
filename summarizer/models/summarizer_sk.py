import numpy as np
from nltk.tokenize import sent_tokenize
from skipthoughts import skipthoughts
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import os
import sys
sys.path.append('skipthoughts')


def load_model():
    print('Loading pre-trained model...')
    global model, encoder
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    print('Encoding sentences...')


def preprocess(docs):
    n_docs = len(docs)
    for i in range(n_docs):
        doc = docs[i].strip()
        lines = doc.split('\n')
        for j in reversed(range(len(lines))):
            lines[j] = lines[j].strip()
            if lines[j] == '':
                lines.pop(j)
        docs[i] = ' '.join(lines)


def split_sentences(docs):
    n_docs = len(docs)
    for i in range(n_docs):
        doc = docs[i]
        sentences = sent_tokenize(doc)
        for j in reversed(range(len(sentences))):
            sent = sentences[j]
            sentences[j] = sent.strip()
            if sent == '':
                sentences.pop(j)
        docs[i] = sentences


def skipthought_encode(docs):
    enc_docs = [None]*len(docs)
    cum_sum_sentences = [0]
    sent_count = 0
    for doc in docs:
        sent_count += len(doc)
        cum_sum_sentences.append(sent_count)

    all_sentences = [sent for doc in docs for sent in doc]
    enc_sentences = encoder.encode(all_sentences, verbose=False)

    for i in range(len(docs)):
        begin = cum_sum_sentences[i]
        end = cum_sum_sentences[i+1]
        enc_docs[i] = enc_sentences[begin:end]
    return enc_docs


def summarize(docs):
    n_docs = len(docs)
    summary = [None]*n_docs
    print('Preprecesing...')
    preprocess(docs)
    print('Splitting into sentences...')
    split_sentences(docs)
    print('Starting to encode...')
    enc_docs = skipthought_encode(docs)
    print('Encoding Finished')
    for i in range(n_docs):
        enc_doc = enc_docs[i]
        n_clusters = int(np.ceil(len(enc_doc)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans = kmeans.fit(enc_doc)
        avg = []
        closest = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, enc_doc)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary[i] = ' '.join([docs[i][closest[idx]] for idx in ordering])
    print('Done')
    return summary
