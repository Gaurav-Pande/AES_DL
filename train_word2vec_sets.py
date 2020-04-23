#@title
# Individual sets
import numpy as np
import warnings
from preprocess import prepare_data
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from baseline_keras import get_model
from preprocess import prepare_data
from utils import *
from visualize import plot_accuracy_curve
warnings.filterwarnings('ignore')
import tensorflow as tf

# Hyperparameters for word2vec
num_features = 200
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3
epochs = 30

# Hyperpaprameters for LSTM
Hidden_dim1=100
Hidden_dim2=64
return_sequences = True
dropout=0.5
recurrent_dropout=0.4
input_size=768
activation='relu'
optimizer = 'adam'
loss_function = 'mean_square_error'
batch_size= 64
epoch =70
model_name = "BiLSTM"
output_dims=1
#####
####
dataset_path='./data/training_set_rel3.tsv'


def train_word2vec_sets():
	set_count = 1
	all_sets_score = []
	data, target, sets = prepare_data(dataset_path=dataset_path)
	for s in sets:
	  print("\n--------SET {}--------\n".format(set_count))
	  set_count +=1
	  X = s
	  y = s['domain1_score']
	  cv = KFold(n_splits=5, shuffle=True)
	  #X, y = prepare_data(dataset_path=dataset_path)
	  cv_data = cv.split(X)
	  results = []
	  prediction_list = []
	  fold_count =1
	  # hyperparameters for word2vec
	  most_common_words= []
	  print(X.shape)
	  print(y.shape)
	  for traincv, testcv in cv_data:
		  print("\n--------Fold {}--------\n".format(fold_count))
		  # get the train and test from the dataset.
		  X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
		  train_essays = X_train['essay']
		  #print("y_train",y_train)
		  test_essays = X_test['essay']
		  #y_train = torch.tensor(y_train,dtype=torch.long)
		  train_sentences = []
		  # print("train_essay ",train_essays.shape)
		  #print(X_train.shape,y_train.shape)
		  for essay in train_essays:
			  # get all the sentences from the essay
			  train_sentences.append(essay_to_wordlist(essay, remove_stopwords = True))

		  # word2vec embedding
		  print("Converting sentences to word2vec model")
		  model,_ = build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
						downsampling)
		  top10 = collections.defaultdict(int)

		  # print("train_sentencesshap",len(train_sentences))
		  trainDataVecs = np.array(getAvgFeatureVecs(train_sentences, model, num_features))
		  test_sentences = []
		  for essay_v in test_essays:
			  test_sentences.append(essay_to_wordlist(essay_v, remove_stopwords=True))
		  testDataVecs = np.array(getAvgFeatureVecs(test_sentences, model, num_features))
		  trainDataVectors = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
		  testDataVectors = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
		  lstm_model = get_model(Hidden_dim1=Hidden_dim1, Hidden_dim2=Hidden_dim2, return_sequences=return_sequences,
								 dropout=dropout, recurrent_dropout=recurrent_dropout, input_size=input_size,
								 activation=activation, model_name=model_name, optimizer=optimizer,
								 loss_function=loss_function)
		  history = lstm_model.fit(trainDataVectors, y_train, batch_size=batch_size, epochs=epoch)

		  # print(trainDataVectors.shape)
		  # print(y_train.shape)
		  history = lstm_model.fit(trainDataVectors, y_train, batch_size=batch_size, epochs=epoch)
		  plot_accuracy_curve(history)
		  y_pred = lstm_model.predict(testDataVectors)
		  y_pred = np.around(y_pred)
		  np.nan_to_num(y_pred)
		  result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
		  print("Kappa Score: {}".format(result))
		  results.append(result)
		  fold_count += 1
		  tf.keras.backend.clear_session()

	  print("Average kappa score value is : {}".format(np.mean(np.asarray(results))))
	  all_sets_score.append(results)
