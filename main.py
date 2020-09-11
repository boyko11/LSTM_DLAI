import numpy as np
from data_utils import *
from model_service import ModelService
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector

if __name__ == '__main__':
    # number of dimensions for the hidden state of each LSTM cell.
    n_a = 64

    X, Y, n_values, indices_values = load_music_utils()
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', n_values)
    print('shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    n_values = 78  # number of music values
    reshapor = Reshape((1, n_values))  # Used in Step 2.B of djmodel(), below
    LSTM_cell = LSTM(n_a, return_state=True)  # Used in Step 2.C
    densor = Dense(n_values, activation='softmax')  # Used in Step 2.D

    model = ModelService.djmodel(Tx=30, n_a=64, n_values=78, reshapor=reshapor, LSTM_cell=LSTM_cell, densor=densor)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))

    model.fit([X, a0, c0], list(Y), epochs=100)

    inference_model = ModelService.music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50)
    inference_model.summary()

    x_initializer = np.zeros((1, 1, 78))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
    print("np.argmax(results[12]) =", np.argmax(results[12]))
    print("np.argmax(results[17]) =", np.argmax(results[17]))
    print("list(indices[12:18]) =", list(indices[12:18]))

    out_stream = generate_music(inference_model)

