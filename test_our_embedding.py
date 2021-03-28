import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import itertools
from collections import Counter
import sklearn
import pickle
from load_model import get_embedding_model

my_sentences = nltk.corpus.gutenberg.sents("austen-emma.txt")
print("num_sentences", len(my_sentences))

n_sentences = 30
with open('word_to_id.pickle', 'rb') as handle:
    word_to_id = pickle.load(handle)
num_idx = len(word_to_id.keys())
embedding = get_embedding_model()

# Remove punctuation
my_sentences = [[w.translate(str.maketrans('', '', string.punctuation)) for w in sent] for sent in my_sentences[:n_sentences]]
# my_sentences = [[w.translate(str.maketrans('', '', string.punctuation)) for w in sent] for sent in my_sentences]
# remove empty strings
my_sentences = [[w for w in sent_words if w != ""] for sent_words in my_sentences]
print("[CORPUS] n_sentences: {}".format(len(my_sentences)))

# Remove stopwords
stop_words = set(stopwords.words('english'))
my_sentences = [[w for w in sent if not w in stop_words] for sent in my_sentences]

# Lemmatize
lemmatizer = WordNetLemmatizer()
my_sentences = [[lemmatizer.lemmatize(w) for w in sent] for sent in my_sentences]
print("[CORPUS] Finished cleaning corpus".format(len(my_sentences)))

# transofrm to words
my_words = list(itertools.chain.from_iterable(my_sentences))
print(my_words[:20])

vocabulary = list(set(my_words))
n_words = len(vocabulary)

# get the most common words
word_counter = Counter(my_words)
n_most_common = 10
most_common_words = word_counter.most_common(n_most_common)
print("most common words:")
print(most_common_words)

# looking each (10-) most common word its similarity with all the others
cos_similarity = np.zeros((n_most_common, n_words))
for i in range(n_most_common):
    print("word1", most_common_words[i][0])
    word1 = np.eye(num_idx)[word_to_id[most_common_words[i][0]]]
    word1 = np.expand_dims(word1, axis=0)
    word1_emb = embedding.predict(word1)
    for j in range(n_words):
        word2 = np.eye(num_idx)[word_to_id[vocabulary[j]]]
        word2 = np.expand_dims(word2, axis=0)
        word2_emb = embedding.predict(word2)
        cos_similarity[i, j] = sklearn.metrics.pairwise.cosine_similarity(word1_emb, word2_emb)

#
max_arg = np.argsort(cos_similarity, axis=1)[:, -5:]
print("shape max_arg", np.shape(max_arg))

for i in range(n_most_common):
    print(most_common_words[i])
    for j in range(5):
        print("idx:", max_arg[i, j], "word:", vocabulary[max_arg[i, j]])
    print()


