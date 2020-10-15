"""
load word2vec learned weights

Michael Stettler and Francesca Capuano
"""
import nltk
from nltk.stem import WordNetLemmatizer 
import string
import pdb 
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle


if __name__ == '__main__':

	"""
	with open('word_to_id.pickle', 'rb') as handle:
	    word_to_id = pickle.load(handle)
	"""



	model = tf.keras.models.load_model("saved_model/my_model", compile=False)
	model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


	print(model.summary())

	model_ouput = model.layers[0].op

	word_embeddings = tf.keras.models.Model(model.input, model_ouput)

	word_embeddings.predict(np.eye(4510)[word_to_id["Jane"]])