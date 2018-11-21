import unicodedata
import re
import tensorflow as tf
import collections
from collections import Counter
# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w

class HeadLineTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, create_vocab=True, vocab_file=None):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        if not create_vocab:
            self.load_vocab(vocab_file)

    def tokenize_headline(self, text):
        split_tokens = []
        for token in preprocess_sentence(text).split(' '):
            split_tokens.append(token)
        return split_tokens 

    def create_vocab(self, train_examples):
        """create a vocabulary from train examples"""
        lang = [preprocess_sentence(example.label) for example in train_examples]
        self.vocab = set()
        vocabcount = Counter(w for txt in lang for w in txt.split())
        vocab = [item[0] for item in vocabcount.most_common()]
        self.vocab = vocab[:10000-3]
        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<end>'] = 2
        self.word2idx['<unkown>'] = 3
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 4            
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a set."""
        with tf.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                self.vocab.update(token)
    def save_vocab(self, vocab_file):
        """Save a vocab set to a txt file"""
        with open(vocab_file, 'w') as f:
            for token in self.vocab:
                f.write(token)
                f.write("\n")
        
    def convert_tokens_to_ids(self, tokens):
        output = []
        for token in tokens:
            if token in self.word2idx:
                output.append(self.word2idx[token])
            else:
                output.append(self.word2idx['<unkown>'])
        return output
    def convert_ids_to_tokens(self, ids):
        output = []
        for id in ids:
            if id in self.idx2word:
                output.append(self.idx2word[id])
            else:
                output.append('<unkown>')
        return output

## stanford output

