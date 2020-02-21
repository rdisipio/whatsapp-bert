from tokenizers import BertWordPieceTokenizer
import numpy as np

class Tokenizer:
    def __init__(self, bert_model = "bert-base-cased"):
        self.bert_model = bert_model
        self.vocabulary_path = "{}-vocab.txt".format(bert_model)
        print("Vocabulary for BERT model {}: {}".format(bert_model, self.vocabulary_path))
        self.tokenizer = BertWordPieceTokenizer(self.vocabulary_path)
    
    def encode(self, plain_text: list, max_length=100):
        token_ids = np.zeros(shape=(len(plain_text), max_length), dtype=np.int32)

        for i, text in enumerate(plain_text):
            encoded = self.tokenizer.encode(text)
            token_ids[i, 0:len(encoded)] = encoded.ids
        attention_masks = (token_ids != 0).astype(np.int32)
        return {"input_ids": token_ids, "attention_masks": attention_masks}

