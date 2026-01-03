import argparse
import os
import pickle
import random
from datetime import datetime
try:
    from tqdm import tqdm
except Exception:
    class SimpleTqdm:
        def __init__(self, iterable, **kw):
            self._iter = iterable
        def __iter__(self):
            return iter(self._iter)
        def __len__(self):
            try:
                return len(self._iter)
            except Exception:
                return 0
        def set_postfix(self, *args, **kwargs):
            return None
    def tqdm(x, **kw):
        return SimpleTqdm(x, **kw)

import numpy as np
from data_loader import load_and_merge_data, load_blocklist
from features import PhishingFeatureExtractor

# Prefer river when available, but fall back to a light sklearn-based incremental wrapper
USE_RIVER = True
try:
    from river import linear_model, preprocessing, metrics
except Exception:
    USE_RIVER = False
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler as SKStandardScaler
    from sklearn import metrics as skmetrics


class OnlineTrainer:
    def __init__(self, model_dir='Trained model', model_name='online_model.pkl', human_in_loop=True,
                 pseudo_label_thresh=0.98, replay_buffer_size=500):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)
        self.human_in_loop = human_in_loop
        self.pseudo_label_thresh = pseudo_label_thresh
        self.replay_buffer_size = replay_buffer_size

        self.blocklist = load_blocklist()
        self.extractor = PhishingFeatureExtractor(blocklist=self.blocklist)

        # river pipeline: scale features then logistic regression
        if USE_RIVER:
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
            self.metric = metrics.ROCAUC()
        else:
            # sklearn incremental fallback (SGDClassifier with log loss)
            self.feature_names = self.extractor.get_feature_names()
            self.scaler = SKStandardScaler()
            self.clf = SGDClassifier(loss='log_loss', max_iter=5)
            self._sk_initialized = False
            # use ROC AUC with sklearn for batch-ish reporting
            self.metric = None

        # simple replay buffer of (x_dict, y)
        self.replay = []

    def _features_to_dict(self, feat_array):
        names = self.extractor.get_feature_names()
        return {n: float(v) for n, v in zip(names, feat_array.tolist())}

    def save(self, versioned=True):
        os.makedirs(self.model_dir, exist_ok=True)
        # save canonical model
        with open(self.model_path, 'wb') as f:
            if USE_RIVER:
                pickle.dump(self.model, f)
            else:
                # save sklearn components together
                pickle.dump({'scaler': self.scaler, 'clf': self.clf, 'feature_names': self.feature_names}, f)
        # also save versioned copy
        if versioned:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            ver_path = os.path.join(self.model_dir, f'online_model_{ts}.pkl')
            with open(ver_path, 'wb') as f:
                if USE_RIVER:
                    pickle.dump(self.model, f)
                else:
                    pickle.dump({'scaler': self.scaler, 'clf': self.clf, 'feature_names': self.feature_names}, f)

    def load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    obj = pickle.load(f)
                    if USE_RIVER:
                        self.model = obj
                    else:
                        # restore sklearn components
                        self.scaler = obj.get('scaler', self.scaler)
                        self.clf = obj.get('clf', self.clf)
                        self.feature_names = obj.get('feature_names', self.feature_names)
                        self._sk_initialized = True
                return True
            except Exception:
                return False
        return False

    def stream_train(self, df, epochs=1, shuffle=True):
        rows = df.to_dict(orient='records')
        for ep in range(epochs):
            if shuffle:
                random.shuffle(rows)
            pbar = tqdm(rows, desc=f'Epoch {ep+1}')
            for r in pbar:
                text = r.get('text', '')
                y = int(r.get('label', 0))
                feats = self.extractor.extract_features(text, return_flags=False)
                x = self._features_to_dict(feats)

                # prediction and metric update
                if USE_RIVER:
                    try:
                        y_pred_proba = self.model.predict_proba_one(x).get(1, 0.0)
                    except Exception:
                        y_pred_proba = 0.5
                    self.metric.update(y, {1: y_pred_proba, 0: 1 - y_pred_proba})
                    # Online supervised update
                    self.model.learn_one(x, y)
                else:
                    # sklearn fallback: convert dict to array
                    X = np.array([x[n] for n in self.feature_names], dtype=float).reshape(1, -1)
                    if not self._sk_initialized:
                        # initialize scaler and classifier
                        self.scaler.partial_fit(X)
                        self.clf.partial_fit(self.scaler.transform(X), [y], classes=[0, 1])
                        self._sk_initialized = True
                        y_pred_proba = 0.5
                    else:
                        try:
                            y_pred_proba = float(self.clf.predict_proba(self.scaler.transform(X))[0][1])
                        except Exception:
                            y_pred_proba = 0.5
                        self.scaler.partial_fit(X)
                        self.clf.partial_fit(self.scaler.transform(X), [y])

                # maintain replay buffer
                self.replay.append((x, y))
                if len(self.replay) > self.replay_buffer_size:
                    self.replay.pop(0)

                if USE_RIVER and self.metric is not None:
                    try:
                        pbar.set_postfix({'roc_auc': f'{self.metric.get():.4f}'})
                    except Exception:
                        pbar.set_postfix({'roc_auc': 'err'})
                else:
                    pbar.set_postfix({'roc_auc': 'n/a'})

            # After each epoch, optionally rehearse replay buffer to reduce forgetting
            if self.replay:
                if USE_RIVER:
                    for x_buf, y_buf in self.replay:
                        self.model.learn_one(x_buf, y_buf)
                else:
                    # rehearse replay for sklearn fallback
                    for x_buf, y_buf in self.replay:
                        X = np.array([x_buf[n] for n in self.feature_names], dtype=float).reshape(1, -1)
                        self.scaler.partial_fit(X)
                        self.clf.partial_fit(self.scaler.transform(X), [y_buf])

        # save after training
        self.save()

    def evaluate(self, df):
        if USE_RIVER:
            metric = metrics.ROCAUC()
            for r in df.to_dict(orient='records'):
                text = r.get('text', '')
                y = int(r.get('label', 0))
                feats = self.extractor.extract_features(text, return_flags=False)
                x = self._features_to_dict(feats)
                try:
                    y_pred_proba = self.model.predict_proba_one(x).get(1, 0.0)
                except Exception:
                    y_pred_proba = 0.5
                metric.update(y, {1: y_pred_proba, 0: 1 - y_pred_proba})
            return metric.get()
        else:
            from sklearn.metrics import roc_auc_score
            y_trues = []
            y_scores = []
            for r in df.to_dict(orient='records'):
                text = r.get('text', '')
                y = int(r.get('label', 0))
                feats = self.extractor.extract_features(text, return_flags=False)
                x = self._features_to_dict(feats)
                X = np.array([x[n] for n in self.feature_names], dtype=float).reshape(1, -1)
                if not self._sk_initialized:
                    y_pred_proba = 0.5
                else:
                    try:
                        y_pred_proba = float(self.clf.predict_proba(self.scaler.transform(X))[0][1])
                    except Exception:
                        y_pred_proba = 0.5
                y_trues.append(y)
                y_scores.append(y_pred_proba)
            try:
                return roc_auc_score(y_trues, y_scores)
            except Exception:
                return 0.5


def main():
    parser = argparse.ArgumentParser(description='Online trainer for phishing classifier')
    parser.add_argument('--epochs', '-e', type=int, default=2)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--model-dir', default='Trained model')
    parser.add_argument('--model-name', default='online_model.pkl')
    parser.add_argument('--replay-size', type=int, default=500)
    parser.add_argument('--resume', action='store_true', help='Load existing model before training')
    args = parser.parse_args()

    print('Loading data...')
    df = load_and_merge_data()
    trainer = OnlineTrainer(model_dir=args.model_dir, model_name=args.model_name,
                            replay_buffer_size=args.replay_size)
    if args.resume:
        loaded = trainer.load()
        print(f'Loaded existing model: {loaded}')

    print('Starting online training (stream)...')
    trainer.stream_train(df, epochs=args.epochs, shuffle=not args.no_shuffle)

    print('Evaluating on full dataset...')
    auc = trainer.evaluate(df)
    print(f'Final ROC-AUC: {auc:.4f}')


if __name__ == '__main__':
    main()
