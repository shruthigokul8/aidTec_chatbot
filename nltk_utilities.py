import nltk
#nltk.download('punkt') #uncomment to download punkt package
from nltk.stem.porter import PorterStemmer
import numpy as np

#initialise stemmer
stemmer = PorterStemmer()

#define tokenisation function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#define stemming function
def stem(word):
    return stemmer.stem(word.lower())

#define bag of words function
def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag =   [  0,     1,      0,   1,     0,      0,       0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag =np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
