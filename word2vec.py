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

def word2vec():
	pass

if __name__ == '__main__':

	# Download Michael, download!
	# nltk.download('gutenberg')
	# Fetch corpus from nltk
	my_words = nltk.corpus.gutenberg.words("austen-emma.txt")
	
	# Remove punctuation	
	my_words = [w.translate(str.maketrans('', '', string.punctuation)) for w in my_words]
	my_words = [w for w in my_words if w != ""]

	# Remove stopwords
	stop_words = set(stopwords.words('english'))   
	my_words = [w for w in my_words if not w in stop_words] 

	# Lemmatize
	lemmatizer = WordNetLemmatizer() 
	my_words = [lemmatizer.lemmatize(w) for w in my_words]	  

	# Create unique id for each token
	word_to_id = {token: idx for idx, token in enumerate(set(my_words))}

	# Convert text to list of word ids
	token_ids = [word_to_id[token] for token in my_words]

	# NEED TO FETCH SENTENCES INSTEAD OF WORDS
	# Training word pairs
	window_size = 2
	data = []
	for i, w in enumerate(my_words[2:5]):
		for j in range(-window_size,window_size+1):
			if j != 0:
				data.append([w, my_words[i + 2 + j]])

	print(data[:20])
	print(my_words[:20])

	data = [[w_centre, w_context] for w_centre in w]

	pdb.set_trace()

	# One hot encodings
	vec = OneHotEncoder()
	X = vec.fit_transform(token_ids)