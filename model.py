
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from tensorflow.keras.models import Model  

def prepare_sequences(train_X, test_X, max_len):
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(train_X) # Only fit on train data
    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    return train_sequences, test_sequences, tokenizer




def build_lstm_model(name, input_dim, embedding_dim, lstm_units, dense_units, max_len, dropout_rate=0.3):
    inputs = Input(shape=(max_len,), name="input_layer") 
    embedding_layer = Embedding(input_dim=input_dim,
                                output_dim=embedding_dim,
                                name="embedding_layer")(inputs)
    lstm_layer = LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, name="lstm_layer")(embedding_layer)
    dense_layer = Dense(dense_units, activation='relu', name="dense_layer")(lstm_layer)
    dropout_layer = Dropout(dropout_rate, name="dropout_layer")(dense_layer)

    outputs = Dense(1, activation='sigmoid', name="output_layer")(dropout_layer) # Either 0 or 1


    model = Model(inputs, outputs, name=name)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    
    model.summary()  
    return model