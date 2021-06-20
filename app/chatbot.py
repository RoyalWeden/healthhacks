import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json

class ChatBot():
    def __init__(self):
        self.words = None
        self.labels = None
        self.training = None
        self.output = None
        self.model = None

    def prepare_intent_data(self):
        with open('app/intents.json') as file:
            self.intents_data = json.load(file)
            
        self.words = []
        self.labels = []
        docs_x = []
        docs_y = []

        for intent in self.intents_data['intents']:
            for pattern in intent['patterns']:
                _words = nltk.word_tokenize(pattern)
                self.words.extend(_words)
                docs_x.append(_words)
                docs_y.append(intent['tag'])

            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])

        self.words = [stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))

        self.labels = sorted(self.labels)


        self.training = []
        self.output = []

        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            _words = [stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in _words:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(docs_y[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)

        self.training = np.array(self.training)
        self.output = np.array(self.output)

    def create_model(self):
        tf.compat.v1.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation='softmax')
        net = tflearn.regression(net)
        
        try:
            self.model = tflearn.DNN(net)
            self.model.load('models/m1.tflearn')
        except:
            self.model = tflearn.DNN(net)
            self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
            self.model.save('models/m1.tflearn')
        finally:
            pass

    def tokenize_words(self, s):
        arr = [0 for _ in range(len(self.words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(self.words):
                if w == se:
                    arr[i] = 1

        return np.array(arr)

    def get_response(self, s):
        result = self.model.predict([self.tokenize_words(s)])
        result_index = np.argmax(result)
        tag = self.labels[result_index]

        for _tag in self.intents_data['intents']:
            if _tag['tag'] == tag:
                responses = _tag['responses']

        return {
            'message': s,
            'all_possible_responses': responses,
            'returned_response': random.choice(responses)
        }