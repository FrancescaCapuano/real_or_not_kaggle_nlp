import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import itertools
from collections import Counter
import spacy
import sklearn

my_sentences = nltk.corpus.gutenberg.sents("austen-emma.txt")
print("num_sentences", len(my_sentences))

n_sentences = 30

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

nlp = spacy.load('en_core_web_md')

cos_similarity = np.zeros((n_most_common, n_words))
for i in range(n_most_common):
    word1 = most_common_words[i][0]
    print("word1", word1)
    word1_emb = nlp(word1)[0].vector
    word1_emb = np.expand_dims(word1_emb, axis=0)
    for j in range(n_words):
        word2 = vocabulary[j]
        word2_emb = nlp(word2)[0].vector
        word2_emb = np.expand_dims(word2_emb, axis=0)
        cos_similarity[i, j] = sklearn.metrics.pairwise.cosine_similarity(word1_emb, word2_emb)

max_arg = np.argsort(cos_similarity, axis=1)[:, -5:]
print("shape max_arg", np.shape(max_arg))

for i in range(n_most_common):
    print(most_common_words[i])
    for j in range(5):
        print("idx:", max_arg[i, j], "word:", vocabulary[max_arg[i, j]])
    print()


