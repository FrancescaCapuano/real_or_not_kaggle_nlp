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

def word2vec():
	pass

if __name__ == '__main__':

	# Download Michael, download!
	# nltk.download('gutenberg')
	# Fetch corpus from nltk
	my_sentences = nltk.corpus.gutenberg.sents("austen-emma.txt")
	
	# Remove punctuation	
	my_sentences = [[w.translate(str.maketrans('', '', string.punctuation)) for w in sent] for sent in my_sentences[:200]]
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

	# Convert text to list of word ids
	sentences_ids = [[word_to_id[token] for token in sent] for sent in my_sentences]

	# Building word pairs
	window_size = 2
	data = []

	for sent in sentences_ids[:3]:
		for i, w in enumerate(sent):
			for j in range(-window_size,window_size+1):
				if j != 0:
					if 0 <=  i+ j < len(sent):
						data.append([w, sent[i + j]])



	# One hot encodings
	n_tokens = len(word_to_id.keys())
	one_hot_encodings = [np.eye(n_tokens)[pair] for pair in data]

	# model creation
	model = keras.Sequential()
	model.add(keras.Input(shape=(n_tokens,)))
	model.add(layers.Dense(300, activation="relu"))
	model.add(layers.Dense(n_tokens))
	
	model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy(),
              metrics=['accuracy'])

	model.fit(train, label)

	
