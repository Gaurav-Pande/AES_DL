import collections
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import preprocess
import multiprocessing
from gensim.models import Word2Vec
import time

top10 = collections.defaultdict(int)

def top10_words(essay_v, remove_stopwords):
    """Get top 10 words from the model."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    top10 = collections.defaultdict(int)
    if remove_stopwords:
        stops = stopwords.words("english")
        ner = preprocess.stop_words()
        stops.extend(ner)
        for word in words:
          if word not in stops:
            # words.append(w)
            top10[word]+=1
    return (top10)

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    #top10 = collections.defaultdict(int)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        for word in words:
          if word not in stops:
            # words.append(w)
            top10[word]+=1
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

def build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                     downsampling):
    model = Word2Vec(workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling)
    # saving the word2vec model
    # model.wv.save_word2vec_format('word2vec_'+ str(fold_count) +'.bin', binary=True)
    cores = multiprocessing.cpu_count()
    print("\n {} cores using".format(cores))
    start_time = time.time()
    model.build_vocab(train_sentences, progress_per=10000)
    print('Time to build vocab using word2vec: {} sec'.format(time.time() - start_time))
    start_time = time.time()
    model.train(train_sentences, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
    print('Time to train the word2vec model: {} mins'.format(time.time() - start_time))
    model.init_sims(replace=True)
    sorted_dic = sorted(top10.items(), key=lambda k: k[1], reverse=True)
    return model,sorted_dic