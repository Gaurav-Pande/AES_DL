from sklearn.model_selection import KFold
from preprocess import prepare_data
from utils import *
from utils import build_word2vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#####
#####
# Hyperparameters for word2vec
num_features = 300
min_word_count = 600
num_workers = 4
context = 10
downsampling = 1e-3
epochs = 30
#####
####
dataset_path='./data/training_set_rel3.tsv'
# x = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

def plot_qwk_scores_all_sets(sets):
  fig = plt.figure()
  ax = plt.subplot(111)
  x = [1,2,3,4,5]
  set1,set2,set3,set4,set5,set6,set7,set8 = sets
  ax.plot(x, set1 , label='set1')
  ax.plot(x, set2, label='set2')
  ax.plot(x, set3, label='set3')
  ax.plot(x, set4, label='set4')
  ax.plot(x, set5, label='set5')
  ax.plot(x, set6, label='set6')
  ax.plot(x, set7, label='set7')
  ax.plot(x, set8, label='set8')
  plt.title('Set wise QWK using BERT for individual sets')
  ax.legend()
  plt.show()

def tsne_plot(model):
	"Creates and TSNE model and plots it"
	labels = []
	tokens = []

	for word in model.wv.vocab:
		tokens.append(model[word])
		labels.append(word)

	tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])

	plt.figure(figsize=(16, 16))
	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i],
					 xy=(x[i], y[i]),
					 xytext=(5, 2),
					 textcoords='offset points',
					 ha='right',
					 va='bottom')
	plt.savefig('Graph.png')
	plt.show()

def plot_accuracy_curve(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['mae'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()


def plot_acrchitecture(filename, model):
	from keras.utils import plot_model
	plot_model(model, to_file=str(filename) + '.png')


def build_visualization():
	cv = KFold(n_splits=2, shuffle=True)
	X, y = prepare_data(dataset_path=dataset_path)
	cv_data = cv.split(X)
	results = []
	prediction_list = []
	fold_count =1
	# hyperparameters for word2vec
	most_common_words= []
	for traincv, testcv in cv_data:
		top10 = collections.defaultdict(int)
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
		model, sorted_dic = build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
								  downsampling)

		for k, v in sorted_dic[:10]:
			print("----------most_similar_word_for:" + str(k) + "--------------")
			print(model.wv.most_similar(k))

		top10 = collections.defaultdict(int)
		tsne_plot(model)



if __name__ == '__main__':
	build_visualization()


