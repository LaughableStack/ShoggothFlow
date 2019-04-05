import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
def get_model(vocab,inputlength):
    model = Sequential()
    model.add(Embedding(vocab, 64, input_length=inputlength))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Dense(vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model