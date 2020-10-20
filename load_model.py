"""
load word2vec learned weights

Michael Stettler and Francesca Capuano
"""
import nltk
from nltk.stem import WordNetLemmatizer 
import string
import pdb
import numpy as np
import tensorflow as tf
import pickle

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def get_embedding_model():
	# load trained model
	model = tf.keras.models.load_model("saved_model/my_model", compile=False)
	model.compile(optimizer='adam',
				  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])

	# keep only first layer
	model_ouput = model.get_layer('dense').output

	# create new model
	return tf.keras.models.Model(model.input, model_ouput)


if __name__ == '__main__':


	with open('word_to_id.pickle', 'rb') as handle:
	    word_to_id = pickle.load(handle)

	num_idx = len(word_to_id.keys())
	print("[DICT] num of id: {}".format(num_idx))
	print("[DICT] jane = {} idx".format(word_to_id["Jane"]))

	# load trained model
	word_embeddings = get_embedding_model()
	print(word_embeddings.summary())

	# encode word to predict
	pred_word = np.eye(num_idx)[word_to_id["Jane"]]
	print("pred word")
	print(pred_word)
	# add dimension to fit the "batch" size required in predict
	pred_word = np.expand_dims(pred_word, axis=0)
	print("shape pred_word", np.shape(pred_word))

	# predict word
	predictions = word_embeddings.predict(pred_word)
	print("shape predictions", np.shape(predictions))
	print(predictions)

