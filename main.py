import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config import *
from features import PhishingFeatureExtractor
from data_loader import load_blocklist, load_and_merge_data
from model_builder import build_hybrid_model


def evaluate_model(model, tokenizer, extractor, scaler, data):
    texts = data['text'].astype(str).values
    labels = data['label'].astype(int).values

    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    feats = np.array([extractor.extract_features(t, return_flags=False) for t in texts])
    feats_scaled = scaler.transform(feats)

    probs = model.predict([padded, feats_scaled], verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'total': len(labels)}


def run_interactive(model, tokenizer, extractor, scaler):
    print("\nPHISHING DETECTOR READY (interactive)")
    while True:
        try:
            user_input = input("Analyze Email > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.strip().lower() == 'exit':
            break

        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        feats, flags = extractor.extract_features(user_input, return_flags=True)
        feats_scaled = scaler.transform([feats])
        prob = float(model.predict([pad, feats_scaled], verbose=0)[0][0])

        override = False
        if flags:
            override = True

        if prob > 0.5 or override:
            print("PHISHING DETECTED")
            print(f"Probability: {prob:.3f}")
            if flags:
                print("Critical flags:")
                for f in flags:
                    print(f" - {f}")
        else:
            print("LEGITIMATE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'serve'], default='serve')
    args = parser.parse_args()

    blocklist = load_blocklist()
    extractor = PhishingFeatureExtractor(blocklist=blocklist)
    scaler = StandardScaler()

    # Prepare data if needed
    data = None
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE) and args.mode != 'train':
        print("Loading saved model and tokenizer...")
        model = tf.keras.models.load_model(MODEL_FILE)
        with open(TOKENIZER_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)

        try:
            data = load_and_merge_data()
        except Exception:
            data = pd.DataFrame({'text': ['dummy'], 'label': [0]})

        texts = data['text'].astype(str).values
        feats_dummy = np.array([extractor.extract_features(t, return_flags=False) for t in texts])
        scaler.fit(feats_dummy)

    if args.mode == 'train':
        print('Training from scratch...')
        try:
            data = load_and_merge_data()
        except Exception as e:
            print('Dataset load failed:', e)
            data = pd.DataFrame({'text': ['Hello friend', 'URGENT: Verify your account now at bit.ly/secure'], 'label': [0, 1]})

        texts = data['text'].astype(str).values
        labels = data['label'].astype(int).values

        tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        os.makedirs(os.path.dirname(TOKENIZER_FILE), exist_ok=True)
        with open(TOKENIZER_FILE, 'wb') as h:
            pickle.dump(tokenizer, h)

        sequences = tokenizer.texts_to_sequences(texts)
        padded_texts = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

        manual_features = np.array([extractor.extract_features(t, return_flags=False) for t in texts])
        manual_features = scaler.fit_transform(manual_features)

        X_text_train, X_test_text, X_feat_train, X_test_feat, y_train, y_test = train_test_split(
            padded_texts, manual_features, labels, test_size=0.2, random_state=42
        )

        model = build_hybrid_model(MAX_VOCAB, MAX_LEN, manual_features.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]

        model.fit([X_text_train, X_feat_train], y_train, epochs=10, batch_size=64,
                  validation_data=([X_test_text, X_test_feat], y_test), callbacks=callbacks)

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        model.save(MODEL_FILE)
        print('Model saved')

        # Evaluate and print confusion counts
        metrics = evaluate_model(model, tokenizer, extractor, scaler, pd.DataFrame({'text': texts, 'label': labels}))
        print('Evaluation on full data:', metrics)

    elif args.mode == 'eval':
        if data is None:
            try:
                data = load_and_merge_data()
            except Exception as e:
                print('Cannot load data to evaluate:', e)
                return
        metrics = evaluate_model(model, tokenizer, extractor, scaler, data)
        print('Evaluation:', metrics)

    else:  # serve interactive
        if data is None:
            # try to load model/tokenizer
            if not (os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE)):
                print('No trained model found. Run with --mode train first.')
                return
        run_interactive(model, tokenizer, extractor, scaler)


if __name__ == '__main__':
    main()
