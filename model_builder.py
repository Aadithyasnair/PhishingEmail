from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Bidirectional
from tensorflow.keras import regularizers
from config import EMBEDDING_DIM


def build_hybrid_model(vocab_size, max_len, num_manual_features):
    # Text branch: embedding + bidirectional LSTM
    text_input = Input(shape=(max_len,), name='text_input')
    x = Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM, mask_zero=True)(text_input)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))(x)
    x = Dropout(0.5)(x)

    # Manual feature branch
    feature_input = Input(shape=(num_manual_features,), name='feature_input')
    y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(feature_input)
    y = Dropout(0.2)(y)

    # Combine
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.4)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[text_input, feature_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
