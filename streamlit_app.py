import os
import re
import pickle
import base64
import requests
import email
from email import policy
import pandas as pd
import numpy as np
import streamlit as st

from config import MODEL_FILE, TOKENIZER_FILE, MAX_LEN
from features import PhishingFeatureExtractor
from data_loader import load_blocklist
from online_trainer import OnlineTrainer, USE_RIVER

st.set_page_config(page_title='Phishing Detector — Live', layout='wide')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEEDBACK_CSV = os.path.join(os.path.dirname(__file__), 'Trained model', 'feedback.csv')


def parse_eml(file_bytes):
    try:
        msg = email.message_from_bytes(file_bytes, policy=policy.default)
        sender = msg.get('From', '')
        subject = msg.get('Subject', '')
        if msg.is_multipart():
            parts = [p for p in msg.walk() if p.get_content_type() == 'text/plain']
            if parts:
                body = parts[0].get_content()
            else:
                body_part = msg.get_body(preferencelist=('plain', 'html'))
                body = body_part.get_content() if body_part else ''
        else:
            body = msg.get_content()
        return sender, subject, body
    except Exception:
        return '', '', ''


@st.cache_resource
def _download_if_needed(url, dst_path):
    try:
        if not url:
            return False, 'no-url'
        if os.path.exists(dst_path):
            return True, 'exists'
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dst_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, 'downloaded'
    except Exception as e:
        return False, str(e)


def load_resources():
    blocklist = load_blocklist()
    extractor = PhishingFeatureExtractor(blocklist=blocklist)

    # load keras model/tokenizer if available
    keras_model = None
    tokenizer = None
    keras_error = None
    # allow model/tokenizer URLs for cloud deploys (set via Streamlit secrets or env vars)
    model_url = os.getenv('MODEL_URL')
    tokenizer_url = os.getenv('TOKENIZER_URL')

    # compute local target paths inside the Update folder
    local_trained_dir = os.path.join(os.path.dirname(__file__), 'Trained model')
    os.makedirs(local_trained_dir, exist_ok=True)
    local_model_path = os.path.join(local_trained_dir, os.path.basename(MODEL_FILE))
    local_tokenizer_path = os.path.join(local_trained_dir, os.path.basename(TOKENIZER_FILE))

    # if URL provided, try download; otherwise fall back to existing configured MODEL_FILE path
    model_to_try = local_model_path if model_url else MODEL_FILE
    tok_to_try = local_tokenizer_path if tokenizer_url else TOKENIZER_FILE

    if model_url:
        ok, msg = _download_if_needed(model_url, local_model_path)
        if not ok:
            keras_error = f"Failed to download model from MODEL_URL: {msg}"

    if tokenizer_url:
        ok_t, msg_t = _download_if_needed(tokenizer_url, local_tokenizer_path)
        if not ok_t:
            # tokenizer download errors are non-fatal; we'll show later
            pass

    try:
        # try loading from the chosen model path
        if os.path.exists(model_to_try):
            import tensorflow as tf
            try:
                keras_model = tf.keras.models.load_model(model_to_try)
            except Exception:
                try:
                    keras_model = tf.keras.models.load_model(model_to_try, compile=False)
                except Exception:
                    raise
    except Exception:
        import traceback
        keras_model = None
        keras_error = traceback.format_exc()

    try:
        if os.path.exists(tok_to_try):
            with open(tok_to_try, 'rb') as f:
                tokenizer = pickle.load(f)
    except Exception:
        tokenizer = None

    trainer = OnlineTrainer(model_dir=os.path.join(os.path.dirname(__file__), 'Trained model'),
                            model_name='online_model.pkl')
    trainer.load()

    return extractor, keras_model, tokenizer, trainer, keras_error


extractor, keras_model, tokenizer, trainer, keras_error = load_resources()


st.title('Phishing Detector — Interactive')
st.markdown('Upload an email (`.eml`) or paste sender/subject/content. Provide feedback to help the model learn.')

with st.expander('Input'):
    eml_file = st.file_uploader('Upload .eml file (optional)', type=['eml'])
    sender_in = st.text_input('Sender (From)')
    subject_in = st.text_input('Subject')
    body_in = st.text_area('Body', height=240)

sender = sender_in
subject = subject_in
body = body_in
if eml_file is not None:
    fb = eml_file.read()
    s, subj, b = parse_eml(fb)
    sender = sender or s
    subject = subject or subj
    body = body or b

combined_text = ''
if sender:
    combined_text += f"From: {sender}\n"
if subject:
    combined_text += f"Subject: {subject}\n"
combined_text += (body or '')

feats = None
flags = None
feat_df = None
keras_prob = None
online_prob = None
feat_names = extractor.get_feature_names()

if combined_text.strip():
    feats, flags = extractor.extract_features(combined_text, sender=sender, subject=subject, return_flags=True)
    feat_df = pd.DataFrame({'feature': feat_names, 'value': feats.tolist()})

    # Keras prediction if available
    if keras_model is not None and tokenizer is not None:
        try:
            seq = tokenizer.texts_to_sequences([combined_text])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
            keras_prob = float(keras_model.predict([pad, np.array([feats])], verbose=0)[0][0])
        except Exception as e:
            keras_prob = None

    # Online trainer prediction (robust for both River and sklearn-saved dicts)
    try:
        x_dict = trainer._features_to_dict(feats) if hasattr(trainer, '_features_to_dict') else {n: float(v) for n, v in zip(feat_names, feats.tolist())}
        # Case A: River pipeline object with predict_proba_one
        if hasattr(trainer, 'model') and not isinstance(trainer.model, dict) and hasattr(trainer.model, 'predict_proba_one'):
            try:
                online_prob = trainer.model.predict_proba_one(x_dict).get(1, 0.0)
            except Exception:
                online_prob = None
        else:
            # Case B: sklearn-style saved dict or trainer attributes
            obj = trainer.model if isinstance(getattr(trainer, 'model', None), dict) else None
            scaler = None
            clf = None
            fn = None
            if obj:
                scaler = obj.get('scaler')
                clf = obj.get('clf')
                fn = obj.get('feature_names')
            else:
                scaler = getattr(trainer, 'scaler', None)
                clf = getattr(trainer, 'clf', None)
                fn = getattr(trainer, 'feature_names', None)
            if clf is not None and scaler is not None and fn is not None:
                X = np.array([x_dict[n] for n in fn], dtype=float).reshape(1, -1)
                try:
                    online_prob = float(clf.predict_proba(scaler.transform(X))[0][1])
                except Exception:
                    online_prob = None
            else:
                online_prob = None
    except Exception:
        online_prob = None

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader('Analysis')
    st.write('Sender:' , sender)
    st.write('Subject:' , subject)
    st.write('Flags:')
    if flags:
        for f in flags:
            st.warning(f)
    else:
        st.success('No critical heuristic flags')
    st.subheader('Feature values')
    if feat_df is not None:
        st.dataframe(feat_df)

with col2:
    st.subheader('Predictions')
    # Keras info or error message
    if keras_model is not None and keras_prob is not None:
        st.metric('Keras Model Probability (phishing)', f'{keras_prob:.3f}')
    else:
        if keras_error:
            st.error('Keras model failed to load; see retrain instructions below')
            with st.expander('View Keras load error'):
                st.code(keras_error)
        else:
            st.info('No Keras model available')
        st.markdown('To train or retrain the Keras model run:')
        st.code('python Update/main.py --mode train')

    if online_prob is not None:
        st.metric('Online Model Probability (phishing)', f'{online_prob:.3f}')
    else:
        st.info('Online incremental model unavailable')

    st.markdown('---')
    st.subheader('Feedback (helps the model learn)')
    feedback = st.radio('Is this a phishing email?', ('Skip / not sure', 'No — Legitimate', 'Yes — Phishing'))
    submit_feedback = st.button('Submit Feedback')

    if submit_feedback and feedback != 'Skip / not sure':
        label = 1 if feedback == 'Yes — Phishing' else 0
        # save feedback to CSV
        os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
        row = {'text': combined_text, 'label': label}
        if os.path.exists(FEEDBACK_CSV):
            df_fb = pd.read_csv(FEEDBACK_CSV)
            df_fb = pd.concat([df_fb, pd.DataFrame([row])], ignore_index=True)
        else:
            df_fb = pd.DataFrame([row])
        df_fb.to_csv(FEEDBACK_CSV, index=False)

        st.success('Feedback saved — updating online model now')

        # online update: train trainer on this single sample
        try:
            trainer.stream_train(df_fb.tail(1), epochs=1, shuffle=False)
            st.success('Online model updated and saved')
        except Exception as e:
            st.error(f'Failed to update online model: {e}')
        # optionally push feedback to GitHub if configured
        gh_token = os.getenv('GITHUB_TOKEN')
        gh_repo = os.getenv('GITHUB_REPO')  # format: owner/repo
        gh_path = os.getenv('GITHUB_FEEDBACK_PATH', 'Update/Trained model/feedback.csv')
        if gh_token and gh_repo:
            try:
                # read file bytes and call GitHub contents API
                import base64, json
                with open(FEEDBACK_CSV, 'rb') as f:
                    content_b64 = base64.b64encode(f.read()).decode('ascii')

                api_url = f'https://api.github.com/repos/{gh_repo}/contents/{gh_path}'
                headers = {'Authorization': f'token {gh_token}', 'Accept': 'application/vnd.github.v3+json'}
                # check if file exists to get sha
                resp = requests.get(api_url, headers=headers)
                if resp.status_code == 200:
                    sha = resp.json().get('sha')
                else:
                    sha = None

                put_payload = {'message': 'Update feedback.csv from Streamlit app', 'content': content_b64}
                if sha:
                    put_payload['sha'] = sha

                r2 = requests.put(api_url, headers=headers, data=json.dumps(put_payload), timeout=20)
                if r2.status_code in (200, 201):
                    st.info('Feedback pushed to GitHub')
                else:
                    st.warning(f'Failed to push feedback to GitHub: {r2.status_code} {r2.text[:200]}')
            except Exception as e:
                st.warning(f'GitHub push failed: {e}')

    st.markdown('---')
    st.subheader('Maintenance')
    if st.button('Retrain Online Model from Stored Feedback'):
        if os.path.exists(FEEDBACK_CSV):
            df_fb = pd.read_csv(FEEDBACK_CSV)
            if not df_fb.empty:
                trainer.stream_train(df_fb, epochs=2)
                st.success('Retraining complete')
            else:
                st.info('No feedback records to train on')
        else:
            st.info('No feedback file found')

    if st.button('Export Latest Online Model (versioned)'):
        try:
            trainer.save(versioned=True)
            st.success('Model saved (versioned copy created)')
        except Exception as e:
            st.error(f'Could not save model: {e}')

    st.markdown('---')
    st.caption('Notes: The app uses a lightweight incremental model for continuous learning. The Keras model (if present) is shown for comparison but is not updated here. Use feedback to improve the incremental model over time.')
