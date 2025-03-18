import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_and_split(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    split_text = df['TEXT'].str.split('[SNIPPET]', expand=True)
    if split_text.shape[1] == 2:
        df[['TEXT1', 'TEXT2']] = split_text
    else:
        print(f"Warning: Some rows could not be split correctly.")
        df['TEXT1'] = split_text[0]
        df['TEXT2'] = split_text[1] if split_text.shape[1] > 1 else None
    df['TEXT1'] = df['TEXT1'].str.strip()
    df['TEXT2'] = df['TEXT2'].str.strip()
    df = df.drop(columns=['TEXT'])
    df = df[['ID', 'TEXT1', 'TEXT2', 'LABEL']]

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['TEXT1'].tolist() + df['TEXT2'].tolist())
    seq1 = tokenizer.texts_to_sequences(df['TEXT1'])
    seq2 = tokenizer.texts_to_sequences(df['TEXT2'])
    max_len = 100
    padded_seq1 = pad_sequences(seq1, maxlen=max_len)
    padded_seq2 = pad_sequences(seq2, maxlen=max_len)

    labels = df['LABEL'].values

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        padded_seq1, padded_seq2, labels, test_size=test_size, random_state=random_state
    )

    return (X1_train, X2_train, y_train), (X1_test, X2_test, y_test), tokenizer, max_len

def train_cnn(X1_train, X2_train, y_train, X1_test, X2_test, y_test):
    max_len = X1_train.shape[1]
    vocab_size = 10000
    embedding_dim = 128

    input1 = Input(shape=(max_len,))
    input2 = Input(shape=(max_len,))

    embedding1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(input1)
    embedding2 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(input2)

    conv1 = Conv1D(64, kernel_size=5, activation='relu')(embedding1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    flat1 = Flatten()(pool1)

    conv2 = Conv1D(64, kernel_size=5, activation='relu')(embedding2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    flat2 = Flatten()(pool2)

    merged = Concatenate()([flat1, flat2])
    dense1 = Dense(128, activation='relu')(merged)
    dropout1 = Dropout(0.5)(dense1)
    output = Dense(1, activation='sigmoid')(dropout1)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit([X1_train, X2_train], y_train, validation_data=([X1_test, X2_test], y_test), epochs=10, batch_size=32)

    print("Training complete.")
    return model

def preprocess_test_data(file_path, tokenizer, max_len):
    df = pd.read_csv(file_path)

    split_text = df['TEXT'].str.split('[SNIPPET]', expand=True)

    if split_text.shape[1] == 2:
        df[['TEXT1', 'TEXT2']] = split_text
    else:
        print(f"Warning: Some rows could not be split correctly.")
        df['TEXT1'] = split_text[0]
        df['TEXT2'] = split_text[1] if split_text.shape[1] > 1 else None

    df['TEXT1'] = df['TEXT1'].str.strip()
    df['TEXT2'] = df['TEXT2'].str.strip()

    if 'LABEL' not in df.columns:
        print("Warning: 'LABEL' column missing in test data.")

    seq1 = tokenizer.texts_to_sequences(df['TEXT1'])
    seq2 = tokenizer.texts_to_sequences(df['TEXT2'])
    padded_seq1 = pad_sequences(seq1, maxlen=max_len)
    padded_seq2 = pad_sequences(seq2, maxlen=max_len)

    return df, padded_seq1, padded_seq2