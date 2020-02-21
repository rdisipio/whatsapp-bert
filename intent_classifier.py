import tensorflow as tf
from transformers import TFBertModel
from tokenizer import Tokenizer
from tensorflow.keras.layers import Dropout, Dense

class IntentClassifier(tf.keras.Model):
    def __init__(self, n_intents=None,
                        dropout=0.2,
                        model_name="bert-base-cased"):
        super().__init__(name="intent_classifier")

        self.tokenizer = Tokenizer()
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

    def get_embedding(self, plain_text, **kwargs):
        encoded = self.tokenizer.encode(plain_text)

        _, pooled_output = self.bert(encoded, **kwargs)
        return pooled_output
