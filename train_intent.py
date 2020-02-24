#!/usr/bin/env python

import os
import pickle
import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from intent_classifier import IntentClassifier
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, data_file_path="intents_db.pkl"):
        self.data = []
        self.known_intents = []
        self.n_intents = 0
        self.intents_labels = {}
        self.intents_embeddings = {}
        self.data_file_path = data_file_path
        # self.model_file_path = "intent_classifier.h5"
        self.encoder = Tokenizer()
        self.model = None

    def initialize(self):
        RELOAD_DATA = os.path.exists(self.data_file_path)
        if RELOAD_DATA:
            logger.info("Reloading data from file {}".format(self.data_file_path))
            with open(self.data_file_path, 'rb') as f:
                self.data = pickle.load(f)
                self.update_intents()
        else:
            self.data = []
            with open("raw_intents.csv", 'r') as f:
                rows = f.readlines()
                for row in rows:
                    if len(row) == 0:
                        continue
                    if row.startswith("#"):
                        continue
                    query, intent = row.split(",")
                    self.data.append({
                        'raw': {
                            'query': query.lower().strip(),
                            'intent': intent.lower().strip()
                        },
                        'nlu':{}
                    })
            self.update_intents()
            self.tokenize_data()
        print("Known intents:")
        print(self.known_intents)

        self.model = IntentClassifier(n_intents=self.n_intents)
        self.model.compile(optimizer='adam', 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['sparse_categorical_accuracy'])
        X, y = self.make_training_dataset(self.data)
        print(X["input_ids"].shape, X['attention_masks'].shape, y.shape)
        self.model.train_on_batch(X, y)
        self.model.summary()
        self.make_intents_embeddings()

    def update_intents(self, new_intent=None):
        if new_intent is None:
            self.known_intents = list(set([x['raw']['intent'] for x in self.data if 'intent' in x['raw']]))
        else:
            if new_intent not in self.known_intents:
                self.known_intents.append(new_intent)
        self.known_intents.sort()
        self.n_intents = len(self.known_intents)
        self.intents_labels = {k:i for i,k in enumerate(self.known_intents)}

    def make_intents_embeddings(self):
        embeddings = self.model.get_embedding(self.known_intents)
        self.intents_embeddings = {k: emb for k, emb in zip(self.known_intents, embeddings) }
        print(self.known_intents)

    def tokenize_data(self):
        query = [x['raw']['query'] for x in self.data]
        encoded = self.encoder.encode(query)
        for i in range(len(self.data)):
            x = self.data[i]
            x['nlu'] = {
                'input_ids':  encoded['input_ids'][i],
                'attention_masks': encoded['attention_masks'][i],
                'label': self.intents_labels[x['raw']['intent']]
            }

    @staticmethod
    def make_training_dataset(batch):
        X = {
            "input_ids": np.array([x['nlu']['input_ids'] for x in batch]),
            'attention_masks': np.array([x['nlu']['attention_masks'] for x in batch])
        }
        y = np.array([x['nlu']['label'] for x in batch], dtype=np.int64)
        return X, y

    def write_out(self):
        # Write out to file
        print("Saving data to file {}".format(self.data_file_path))
        with open(self.data_file_path, 'wb') as f:
            pickle.dump(self.data, f)
        #model.save(model_file_path)
    
    def predict_intent(self, txt):
        this_embedding = self.model.get_embedding([txt])
        all_embeddings = [self.intents_embeddings[i] for i in self.known_intents]
        scores = cosine_similarity(this_embedding, all_embeddings)
        k = np.argmax(scores[0])
        confidence = scores[0][k]
        closest_intent = self.known_intents[k]
        return closest_intent, confidence

    def loop(self):
        while True:
            print("Tell me what you would like to do")
            txt = input()
            txt = txt.lower()
            if txt in ['q', 'quit', 'stop']:
                return

            intent, confidence = self.predict_intent(txt)
            print("Is this your purpose? {} (confidence={:.3f})".format(intent, confidence))
            reply = input().lower()
            if reply in ['n', 'no', 'nope']:    
                print("What is the purpose?")
                intent = input().lower()
                if intent not in self.known_intents:
                    closest_intent, confidence = self.predict_intent(intent)
                    print("Is this the same as {} (confidence={:.3f})? [y, n]".format(closest_intent, confidence))
                    reply = input().lower()
                    if reply in ['y', 'yes']:
                        intent = closest_intent
                    else:
                        print("This is a new intent to me")
                        self.update_intents(intent)
                        self.make_intents_embeddings()
            print("Ok, so you are asking for: {}".format(intent))

            encoded = self.encoder.encode(txt)
            entry = {
                'raw': {
                    'query': txt,
                    'intent': intent
                },
                'nlu': {
                    'input_ids': encoded['input_ids'][0],
                    'attention_masks': encoded['attention_masks'][0],
                    'label': self.intents_labels[intent]
                }
            }
            self.data.append(entry)

            X, y = self.make_training_dataset([entry])
            self.model.train_on_batch(X, y)


if __name__ == '__main__':
    engine = Engine()
    engine.initialize()
    engine.loop()
    engine.write_out()