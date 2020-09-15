import os
import sys
import requests
import zipfile
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.models import save_model
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def get_init_epoch(checkpoint_path):
    return int(checkpoint_path.split('.')[2].split('-')[0])

data_dir = f'./data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

glove_dir = f'./glove'
if not os.path.exists(glove_dir):
    os.makedirs(glove_dir, exist_ok=True)

temp = False
for file in os.listdir(glove_dir):
    if file == 'glove.840B.300d.txt':
        temp = True
        break

if temp == False:
    print('Downloading glove is started ...')
    url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    r = requests.get(url, allow_redirects=True)
    with open(os.path.join(glove_dir, 'glove.840B.300d.zip'), 'wb') as f:
        f.write(r.content)
    print('Downloading glove is completed ...')
    with zipfile.ZipFile(os.path.join(glove_dir, 'glove.840B.300d.zip'), 'r') as zip_ref:
        zip_ref.extractall(glove_dir)

save_dir = f'./checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

max_val = 0
checkpoint = None
for file in os.listdir(save_dir):
    if file.endswith('.hdf5'):
        if file.split('.')[0] == sys.argv[0].split('.')[0]:
            if get_init_epoch(os.path.join(save_dir, file)) > max_val:
                checkpoint = os.path.join(save_dir, file)
                max_val = get_init_epoch(os.path.join(save_dir, file))

print('Checkpoint path: {}'.format(checkpoint))

# Import data and labels
xdata = [line.rstrip('\n') for line in open("./data/xtrain_obfuscated.txt")]
ydata = [int(line.rstrip('\n')) for line in open("./data/ytrain.txt")]

# Split (training) data into training and validation
xtrain, xvalid, ytrain, yvalid = train_test_split(xdata, ydata, test_size=0.20)

# Parsing the GloVe word-embedding file
embeddings_index = {}
f = open('./glove/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype=np.float32)
    except ValueError:
        continue
    embeddings_index[word] = coefs
f.close()

# Using Keras tokenizer
max_len = 450
token = text.Tokenizer(num_words=None, char_level=True)
token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

# Zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

# Preparing the GloVe word-embeddings matrix
embedding_dim = 300
embedding_matrix = np.zeros((len(token.word_index) + 1, embedding_dim))
for word, i in tqdm(token.word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Binarizing labels for the neural network
ytrain_enc = np_utils.to_categorical((ytrain))
yvalid_enc = np_utils.to_categorical((yvalid))

# Model definition
# Bidirectional LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(token.word_index) + 1,
                     embedding_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(12))
model.add(Activation('softmax'))
model.summary()

file_name = os.path.basename(sys.argv[0]).split('.')[0]

check_cb = ModelCheckpoint('./checkpoints/' + file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

earlystop_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# Training and evaluation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Loading checkpoint
if checkpoint is not None:
    # Loading the model from checkpoint
    model.load_weights(checkpoint)
    # Finding the epoch index from which we are resuming
    initial_epoch = get_init_epoch(checkpoint)
else:
    initial_epoch = 0

model.fit(xtrain_pad, y=ytrain_enc, validation_data=(xvalid_pad, yvalid_enc), batch_size=64, epochs=200, verbose=1, shuffle=True, callbacks=[check_cb, earlystop_cb], initial_epoch=initial_epoch)

# Saving the final model
file_name = os.path.basename(sys.argv[0]).split('.')[0]
save_model(model,'./checkpoints/' + file_name + '_final.hdf5')
