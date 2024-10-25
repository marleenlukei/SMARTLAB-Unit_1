
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model  

def prepare_sequences(train_X, test_X, max_len):
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(train_X) # Only fit on train data
    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    return train_sequences, test_sequences, tokenizer




def build_model(name, input_dim, embedding_dim, lstm_units, dense_units, unstructured_len, structured_len):
    unstructured_input = Input(shape=(unstructured_len,))
    embedding_layer = Embedding(input_dim=input_dim, output_dim=embedding_dim)(unstructured_input)
    lstm_out = LSTM(lstm_units)(embedding_layer)

    # Structured input branch
    structured_input = Input(shape=(structured_len,))
    structured_dense = Dense(dense_units, activation='relu')(structured_input)

    # Concatenate text and structured branches
    concatenated = Concatenate()([lstm_out, structured_dense])
    output = Dense(1, activation='sigmoid')(concatenated)  # For binary classification

    # Build and compile the model
    model = Model(inputs=[unstructured_input, structured_input], outputs=output, name=name)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model