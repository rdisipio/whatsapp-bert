import tensorflow as tf
from transformers import TFBertModel

from tensorflow.keras.layers import Dropout, Dense

class IntentClassifier():
    def __init__(self, n_intents,
                        dropout=0.2,
                        model_name="bert-base-cased"):
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout)
        self.intent_classifier = Dense(n_intents, activation='softmax')
    
    def call(self, inputs, **kwargs):
        # The second output of the main BERT layer corresponds to the [CLS] token
        # and gives a pooled representation for the full sequence

        sequence_output, pooled_output = self.bert(inputs, **kwargs)
        pooled_output = self.dropout(pooled_output)
        intent = self.intent_classifier(pooled_output)
        return intent