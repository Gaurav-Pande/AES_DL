# AES_DL
The code in this section is exploration of different deep learning techniques on individual sets and on whole dataset.

### Little bit about Data:

we used the Auto-mated Student Assessment Prize (ASAP) datasetby The Hewlett Foundation.  (Hewlett, 2012:  ac-cessed March 12, 2020) 
This dataset consists ofessays written by students from 7th - 10th grade.The essays are divided into 8 sets.  Each set hasa prompt associated with it.  There are 2 types ofprompt Type 1: Persuasive / Narrative / ExpositoryType 2:  Source Dependent Responses.  The firsttype of prompt asks students to state their opinionabout certain topic.  The second type of prompthas a required reading associated with it and thestudents are expected to answer a question basedon their understanding of this reading.  Differentprompts have been graded by different number ofgraders. But each set has a domain 1 score, which
 
 
### So what did we try:
 
The approaches tried in DL are:
1. Try 3 different architecture involving LSTM, BiLSTM, and CNNs on individual sets and on whole 
dataset separately. 
2. Use Word2vec and Bert embeddings for feature vector representation.
3. Hyperparameter tunning to optimize the loss and increase the mean QWK.
 
Currently the models were trained in keras(tensorflow as backend).


### Prerequisites

* Python 3+
* compute as it takes around 4-5 hours to run all the models and approaches.


### Installation

I would recommend using google collab or better if you have GPU access. If you are running this locally then
follow the instructions:

* Install virtual environment using:

```shell script
pip install virtualenv
```

* Create a virtual environment using:

```shell script

virtualenv aes

```

* Activate virtual environment

```shell script

source aes/bin/activate

```

* Install requirements from requirements.txt

```shell script

pip install -r requirements.txt

```

#####  Training the models

* To train the model using BERT, first change the hyperparameters in the train_{BERT/word2vec}_{sets/all}.py file
* once you have changed the hyperparameters, run the respective file for training. For example

Using BERT and train on per set, run:

```shell script
python train_bert_sets.py
```  

*  Using BERT and train on whole data set, run:

```shell script
python train_bert_all.py
```  



*  Using WORD2VEC and train on whole dataset, run:

```shell script

python train_word2vec_all.py

```  


#### Note:

* If you would like to run the Notebook then you can directly open the AES.ipynb, and run cell by cell,
but the results there may vary.
* There is a PDF of the Notebook attached to see the results we got after training the models.


[Future Work]:

* add commmand line parameters for passing hyperparameters.
* add pytorch support.
* add GPU support for models.
* add more extensive hyperparameters.
* add sigmoid activation.
* Topic modelling using LDA.
* Visualization for topic modelling.




