# [TODO]: This is under developement
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicLSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(BasicLSTM, self).__init__()
		# hidden dimension ==> each context length
		self.hidden_dim = hidden_dim
		# Embedding layer
		self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
		# LSTM layer
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)
		# feed forward layer
		self.hiddenTotag = nn.Linear(hidden_dim, tagset_size)

	def forward(self, sentence):
		tag_scores = None
		ebdngs = self.word_embedding(sentence)
		lstm_output, _ = self.lstm(ebdngs.view(len(sentence), 1, -1))
		tag_seq = self.hiddenTotag(lstm_output.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_seq, dim=1)
		return tag_scores