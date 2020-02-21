#!/usr/bin/env python

from intent_classifier import IntentClassifier

known_intents = [

]
n_intents = len(known_intents)

if __name__ == '__main__':
    model = IntentClassifier(n_intents=n_intents)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
