import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print("train_df", np.shape(train_df))
print("test_df", np.shape(test_df))

train_df[train_df["target"] == 0]["text"].values[1]

# print first sentence (should be 13 words)
print(train_df.iloc[0]["text"])
print(train_df.iloc[2]["text"])

# building vectors
count_vectorizer = feature_extraction.text.CountVectorizer()

## let's get counts for training set
example_train_vectors = count_vectorizer.fit_transform(train_df["text"])
print("example_train_vectors", np.shape(example_train_vectors))  # num_tweets, vocabulary

# create train_vectors as vectors of counts of unique words (vocabulary)
train_vectors = count_vectorizer.fit_transform(train_df["text"])
print("shape train_vectors", np.shape(train_vectors))

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors -
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])

# todo
# 1) check if concatenate or adding up embedded vectors
# 2) look at back propagation tutorial
