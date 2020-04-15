import time
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from AES_DL.baseline_keras import get_model
from AES_DL.preprocess import prepare_data
from AES_DL.utils import *
import multiprocessing
from gensim.models import Word2Vec


#####
#####
# Hyperparameters for word2vec
num_features = 300
min_word_count = 600
num_workers = 4
context = 10
downsampling = 1e-3
epochs = 30

# Hyperpaprameters for LSTM
Hidden_dim1=300
Hidden_dim2=64
return_sequences = True
dropout=0.5
recurrent_dropout=0.4
input_size=768
activation='relu'
bidirectional = False
batch_size = 64
epoch = 70
#####
####
dataset_path='./data/training_set_rel3.tsv'



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



def train_word2vec():
    cv = KFold(n_splits=2, shuffle=True)
    X, y = prepare_data(dataset_path=dataset_path)
    cv_data = cv.split(X)
    results = []
    prediction_list = []
    fold_count =1
    # hyperparameters for word2vec
    most_common_words= []
    for traincv, testcv in cv_data:

        print("\n--------Fold {}--------\n".format(fold_count))
        # get the train and test from the dataset.
        X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
        train_essays = X_train['essay']
        #print("y_train",y_train)
        test_essays = X_test['essay']
        #y_train = torch.tensor(y_train,dtype=torch.long)
        train_sentences = []

        for essay in train_essays:
            # get all the sentences from the essay
            train_sentences += essay_to_sentences(essay, remove_stopwords = True)

        # word2vec embedding
        print("Converting sentences to word2vec model")
        model,_ = build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                     downsampling)
        top10 = collections.defaultdict(int)


        trainDataVecs = np.array(getAvgFeatureVecs(train_sentences, model, num_features))
        test_sentences = []
        for essay_v in test_essays:
            test_sentences.append(essay_to_wordlist(essay_v, remove_stopwords=True))
        testDataVecs = np.array(getAvgFeatureVecs(test_sentences, model, num_features))
        trainDataVectors = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
        testDataVectors = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
        lstm_model = get_model(Hidden_dim1=Hidden_dim1, Hidden_dim2=Hidden_dim2, return_sequences=return_sequences,
                               dropout=dropout, recurrent_dropout=recurrent_dropout, input_size=input_size,
                               activation=activation, bidirectional=bidirectional)
        lstm_model.fit(trainDataVectors, y_train, batch_size=batch_size, epochs=epoch)
        y_pred = lstm_model.predict(testDataVectors)
        y_pred = np.around(y_pred)
        np.nan_to_num(y_pred)
        result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
        print("Kappa Score: {}".format(result))
        results.append(result)
        fold_count += 1

    print("Average kappa score value is : {}".format(np.mean(np.asarray(results))))


if __name__ == '__main__':
    train_word2vec()