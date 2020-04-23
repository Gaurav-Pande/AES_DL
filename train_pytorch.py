# [TODO]: This is under developement

import torch
def train(epoch, model, loss_function, optimizer, training_data, y):
	train_loss = 0
	train_examples = 0
	print(y)
	for x_sentence, y_sentence in zip(training_data, y):
		model.zero_grad()
		# sen_input = prepare_sequence(sentence,word_to_idx)
		# targets = prepare_sequence(tags,tag_to_idx)
		# print(targets.shape)
		print(x_sentence.shape)
		tag_scores = model(x_sentence)
		y_sentence = y_sentence.unsqueeze(0)
		loss = loss_function(tag_scores, y_sentence)
		train_examples += 1
		loss.backward()
		optimizer.step()
		train_loss += loss
		train_examples += len(y)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	avg_train_loss = train_loss / train_examples
	avg_val_loss, val_accuracy = evaluate(model, loss_function, optimizer)

	print("Epoch: {}/{}\tAvg Train Loss: {:.4f}\tAvg Val Loss: {:.4f}\t Val Accuracy: {:.0f}".format(epoch,

	                                                                                                 avg_train_loss,
	                                                                                                 avg_val_loss,
	                                                                                                 val_accuracy))


def evaluate(model, loss_function, optimizer, val_data):
	# returns:: avg_val_loss (float)
	# returns:: val_accuracy (float)
	val_loss = 0
	correct = 0
	val_examples = 0
	with torch.no_grad():
		for sentence, tags in val_data:
			#############################################################################
			# TODO: Implement the evaluate loop
			# Find the average validation loss along with the validation accuracy.
			# Hint: To find the accuracy, argmax of tag predictions can be used.
			#############################################################################
			# model.zero_grad()
			# # sen_input = prepare_sequence(sentence, word_to_idx)
			# # targets = prepare_sequence(tags, tag_to_idx)
			# tag_scores = model(sen_input)
			# _, indices = torch.max(tag_scores, 1)
			# val_loss += loss_function(tag_scores, targets)
			# correct += torch.sum(indices == torch.LongTensor(targets))
			# val_examples += len(targets)
			passs

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
	val_accuracy = 100. * correct / val_examples
	avg_val_loss = val_loss / val_examples
	return avg_val_loss, val_accuracy
