from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM

DROPOUT = 0.2
INTER_DIM = (20, 10)
INPUT_LENGTH = 20
INPUT_DIM = 17
OUTPUT_DIM = 3

def build_model():
    model = Sequential()

    model.add(LSTM(INTER_DIM[0], return_sequences=True, dropout_W=DROPOUT, dropout_U=DROPOUT,
                   input_length=INPUT_LENGTH, input_dim=INPUT_DIM))
    model.add(LSTM(INTER_DIM[1], return_sequences=True, dropout_W=DROPOUT, dropout_U=DROPOUT))
    model.add(TimeDistributed(Dense(OUTPUT_DIM)))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model