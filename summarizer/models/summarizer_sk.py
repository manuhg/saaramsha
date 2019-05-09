import numpy as np
from nltk.tokenize import sent_tokenize
from skipthoughts import skipthoughts
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from attn import load_ed,eval_ed
import os
import sys
sys.path.append('skipthoughts')

def reduction_scheme_len(doc_len,scheme='half'):
  schemes = {'half':lambda n:int(np.ceil(float(n)*0.5)),
             'onethird':lambda n:int(np.ceil(float(n)/3)),'onefourth':lambda n:int(np.ceil(float(n)/4)),
             'sqrt':lambda n:int(np.ceil(float(n)**0.5))}
  return schemes.get(scheme)(doc_len)

def getval(dfg,i,index=True):
    if index:
        return dfg.loc[i,:][1].index.tolist()
    return list(dfg.loc[i,:][1].loc[:,'doc'])

def get_clusters(doc,rsi=0,index=True):#rsi - reduction sceheme index
    reduction_schemes = ['half','onethird','onefourth','sqrt',]
    random_state = 0
    n_clusters = max([2,reduction_scheme_len(len(doc),reduction_schemes[rsi])])
    print('Number of clusters =',n_clusters)
    km = KMeans(n_clusters=n_clusters,random_state=random_state)
    kmc = km.fit(doc)
    dc = [e.tolist() for e in doc ]
    df = pd.DataFrame({'doc':dc,'cluster':kmc.predict(doc)})
    dfg = pd.DataFrame(df.groupby(by=['cluster']))
    clusters = list(map( lambda i:getval(dfg,i,index),list(range(len(dfg)))))
    return clusters

def load_model():
    print('Loading pre-trained model...')
    global model, encoder
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    try:
        load_ed()
    except Exception as e:
        pass
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

def dec_sum(enc_doc,doc):
    sdoc = split_sentences([doc])[0]
    clusters_alt = get_clusters(enc_doc,rsi=0,index=True)
    sent_clusters = list(map(lambda cluster: list(map(lambda index: sdoc[index],cluster)),clusters_alt))
    dec_summ = []
    for s in sent_clusters:
        dec_summ.append(eval_ed(s))
    return dec_summ


def summarize(docs):
    n_docs = len(docs)
    summary_ed =[]
    summary = [None]*n_docs
    print('Preprecesing...')
    preprocess(docs)
    print('Splitting into sentences...')
    split_sentences(docs)
    print('Starting to encode...')
    enc_docs = skipthought_encode(docs)
    print('Encoding Finished')
    tsne_components = 512
    for i in range(n_docs):
        enc_doc = enc_docs[i]
        enc_doc = TSNE(n_components=tsne_components).fit_transform(enc_doc)
        n_clusters = int(reduction_scheme_len(len(docs[i])))
        n_clusters = max([1,reduction_scheme_len(len(docs[i]),'sqrt')])
        
        
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

        summary_ed.append(dec_sum(enc_docs[i],docs[i]))
    print('Done')
    
    return summary
