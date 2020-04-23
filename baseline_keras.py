from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Input, Bidirectional,Conv2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential, load_model, model_from_config, Model
import keras.backend as K

def get_model(Hidden_dim1=300, Hidden_dim2=64, return_sequences = True, dropout=0.5,
              recurrent_dropout=0.4, input_size=768, activation='relu', loss_function = 'mean_squared_error',
              optimizer= "adam", model_name = "BiLSTM", output_dims = 1):
    """Define the model."""
    model = Sequential()
    if model_name == 'BiLSTM':
        model.add(Bidirectional(LSTM(Hidden_dim1,return_sequences=return_sequences , dropout=0.4, recurrent_dropout=recurrent_dropout), input_shape=[1, input_size]))
        model.add(Bidirectional(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout)))
    if model_name == "LSTM":
        model.add(LSTM(Hidden_dim1, dropout=0.4, recurrent_dropout=recurrent_dropout, input_shape=[1, input_size], return_sequences=return_sequences))
        model.add(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout))
    if model_name == "CNN":
        inputs = Input(shape=(768, 1))
        x = Conv1D(64, 3, strides=1, padding='same', activation=activation)(inputs)
        # Cuts the size of the output in half, maxing over every 2 inputs
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation=activation)(x)
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(output_dims, activation=activation)(x)
        model = Model(inputs=inputs, outputs=outputs, name='CNN')
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model





