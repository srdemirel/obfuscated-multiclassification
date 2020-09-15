import os
import sys
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence, text

def get_init_epoch(checkpoint_path):
    return int(checkpoint_path.split('.')[2].split('-')[0])

results_dir = f'./results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

save_dir = f'./checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Importing test data
xtest = [line.rstrip('\n') for line in open("./data/xtest_obfuscated.txt")]

# Using Keras tokenizer
max_len = 450
token = text.Tokenizer(num_words=None, char_level=True)
token.fit_on_texts(list(xtest))
xtest_seq = token.texts_to_sequences(xtest)

# Zero pad the sequences
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

# Loading the model
model = load_model('./checkpoints/convnet_with_glove_emb.97-0.76.hdf5')
# model = load_model('./checkpoints/lstm_with_glove_emb.98-0.84.hdf5')

# Obtaining the predictions
predictions = model.predict_classes(xtest_pad)

# Writing the result in a txt file
with open(os.path.join(results_dir, 'ytest.txt'),'w') as result:
    for idx in range(len(predictions)):
        result.write(str(predictions[idx]) + os.linesep)
