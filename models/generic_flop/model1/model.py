from keras.models import Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM, merge, Input

BATCH_SIZE = 500
NUM_EPOCHS = 5
INPUT_DIM = 17
FLOP_DIM = 42
OUTPUT_DIM = 3
INPUT_LENGTH = 20
INTER_DIM = (30, 10)
FLOP_INTER_DIM = (30, 20, 10)
DROPOUT = 0.2

def build_model():
    action_input = Input(shape=(INPUT_LENGTH, INPUT_DIM))
    actual_flop_input = Input(shape=(INPUT_LENGTH, FLOP_DIM))
    flop_input = actual_flop_input

    # 2 dense layers to encode flop
    for dim in FLOP_INTER_DIM:
        flop_input = TimeDistributed(Dense(dim))(flop_input)

    seq = merge([action_input, flop_input], mode='concat', concat_axis=2)
    
    for dim in INTER_DIM:
        seq = LSTM(dim, return_sequences=True, dropout_W=DROPOUT, dropout_U=DROPOUT)(seq)
    seq = TimeDistributed(Dense(OUTPUT_DIM))(seq)
    probs = Activation('softmax')(seq)

    model = Model(input=[action_input, actual_flop_input], output=probs)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model