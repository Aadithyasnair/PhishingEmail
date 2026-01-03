import os

# ========================================================
# CONFIGURATION & DATASETS (paths resolved relative to this file)
# ========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_FILES = [
    os.path.join(BASE_DIR, "Datasets", "spam_ham_dataset.csv"),
    os.path.join(BASE_DIR, "Datasets", "nlp_dataset.csv"),
    os.path.join(BASE_DIR, "Datasets", "Phishing_validation_emails.csv"),
    os.path.join(BASE_DIR, "Datasets", "Phishing_Email.csv"),
    os.path.join(BASE_DIR, "Datasets", "Phishing_Legit_Email.csv"),
    os.path.join(BASE_DIR, "Datasets", "CEAS_08.csv"),
    os.path.join(BASE_DIR, "Datasets", "Enron.csv"),
]

BLOCKLIST_FILE = os.path.join(BASE_DIR, "Text files", "known_phishing_urls.txt")
MODEL_FILE = os.path.join(BASE_DIR, "Trained model", "phishing_model.keras")
TOKENIZER_FILE = os.path.join(BASE_DIR, "Trained model", "tokenizer.pickle")

MAX_VOCAB = 10000
MAX_LEN = 150
EMBEDDING_DIM = 100
