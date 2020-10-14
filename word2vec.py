"""
implementation of word2vec algorithm in tensorflow

Michael Stettler and Francesca Capuano
"""
import nltk
from nltk.stem import WordNetLemmatizer 
import string
import pdb 
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

def word2vec():
	pass

if __name__ == '__main__':

	# Download Michael, download!
	# nltk.download('gutenberg')
	# Fetch corpus from nltk
	my_sentences = nltk.corpus.gutenberg.sents("austen-emma.txt")

	n_sentences = 3000
	
	# Remove punctuation	
	my_sentences = [[w.translate(str.maketrans('', '', string.punctuation)) for w in sent] for sent in my_sentences[:n_sentences]]
	# remove empty strings
	my_sentences = [[w for w in sent_words if w != ""] for sent_words in my_sentences]

	# Remove stopwords
	stop_words = set(stopwords.words('english'))   
	my_sentences = [[w for w in sent if not w in stop_words] for sent in my_sentences]

	# Lemmatize
	lemmatizer = WordNetLemmatizer() 
	my_sentences = [[lemmatizer.lemmatize(w) for w in sent] for sent in my_sentences]

	# Create unique id for each token
	word_to_id = {token: idx for idx, token in enumerate(set([w for sent in my_sentences for w in sent]))}

	# Save word to id dictionary
	with open('word_to_id.pickle', 'wb') as handle:
	    pickle.dump(word_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print(len(word_to_id.keys()))

	print(word_to_id["Jane"])
	pdb.set_trace()

	# Convert text to list of word ids
	sentences_ids = [[word_to_id[token] for token in sent] for sent in my_sentences]

	# Building word pairs
	window_size = 2
	data = []



	for sent in sentences_ids:
		for i, w in enumerate(sent):
			for j in range(-window_size,window_size+1):
				if j != 0:
					if 0 <=  i+ j < len(sent):
						data.append([w, sent[i + j]])



	# One hot encodings
	n_tokens = len(word_to_id.keys())
	one_hot_encodings = [np.eye(n_tokens)[pair] for pair in data]

	train = np.array(one_hot_encodings)[:, 0, :]
	label = np.array(one_hot_encodings)[:, 1, :]




	# model creation
	inputs = keras.Input(shape=(n_tokens,))
	x = keras.layers.Dense(300, activation="relu")(inputs)
	outputs = keras.layers.Dense(n_tokens)(x)
	model = keras.models.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	model.fit(train, label, epochs=100)