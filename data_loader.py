import os
import pandas as pd
from config import DATASET_FILES, BLOCKLIST_FILE

def load_blocklist():
    """Loads the URL blocklist if exists."""
    blocklist = set()
    if os.path.exists(BLOCKLIST_FILE):
        try:
            with open(BLOCKLIST_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line: 
                        blocklist.add(line)
            print(f"✅ Loaded Blocklist ({len(blocklist)} URLs)")
        except Exception as e:
            print(f"⚠️ Error loading blocklist: {e}")
    return blocklist

def load_and_merge_data():
    all_dfs = []
    print("Loading datasets...")
    
    for file in DATASET_FILES:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
                cols_lower = {c.lower(): c for c in df.columns}
                
                text_candidates = ['text', 'content', 'body', 'message', 'email', 'email text']
                text_col = next((cols_lower[t] for t in text_candidates if t in cols_lower), None)
                
                label_candidates = ['label', 'class', 'phishing', 'type', 'email type']
                label_col = next((cols_lower[t] for t in label_candidates if t in cols_lower), None)
                
                if text_col and label_col:
                    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
                    # Robust label mapping
                    def map_label(x):
                        if pd.isna(x):
                            return None
                        s = str(x).lower().strip()
                        if s in ('ham', '0', 'safe', 'legit', 'safe email', 'not phishing', 'non-phishing'):
                            return 0
                        if s in ('spam', '1', 'phishing', 'fraud', 'phishing email', 'scam'):
                            return 1
                        # try numeric
                        try:
                            v = int(s)
                            return 1 if v == 1 else 0
                        except Exception:
                            return None

                    df['label'] = df['label'].apply(map_label)
                    df = df.dropna()
                    all_dfs.append(df)
                    print(f"✅ Loaded {file} ({len(df)} rows)")
                else:
                    print(f"⚠️  Skipping {file}: Could not identify text/label columns.")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    if not all_dfs:
        raise ValueError("No datasets loaded! Please ensure .csv files are in the directory.")

    return pd.concat(all_dfs, ignore_index=True)
